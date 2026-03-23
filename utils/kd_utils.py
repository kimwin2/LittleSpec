import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class KDTrainer(Trainer):
    def __init__(self, teacher_model, l2l_loss_scale, *args, **kwargs):
        super(KDTrainer, self).__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        # self.teacher_model = self.teacher_model.eval()
        self.l2l_loss_scale = l2l_loss_scale

    def ce_loss(self, student_logits, teacher_logits):
        model_output_log_prob = F.log_softmax(student_logits, dim=-1)
        real_output_soft = F.softmax(teacher_logits, dim=-1)

        return F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")

    def mse_loss(self, student_logits, teacher_logits):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return F.mse_loss(student_logits, teacher_logits)

    # Implement KD functions
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["output_hidden_states"] = True

        # Teacher Model Inference (KD)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        teacher_logits = teacher_outputs.get("logits")
        teacher_reps = teacher_outputs.hidden_states[1:]
        del teacher_outputs

        # Student Model Inference
        outputs = model(**inputs)

        student_logits = outputs.get("logits")
        student_reps = outputs.hidden_states[1:]

        if not return_outputs:
            del outputs

        kd_loss = self.ce_loss(student_logits, teacher_logits)

        l2l_loss = 0
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            tmp_loss = self.mse_loss(student_rep, teacher_rep)
            l2l_loss += tmp_loss
        l2l_loss = self.l2l_loss_scale * l2l_loss

        loss = kd_loss + l2l_loss

        self.log({
            "l2l_loss": l2l_loss.item(),
            "kd_loss": kd_loss.item(),
        })

        return (loss, outputs) if return_outputs else loss


class TrainTimeTestKDTrainer(KDTrainer):
    """
    KDTrainer with EAGLE-style training-time test (multi-step autoregressive rollout).
    
    During training, the student model predicts not just the next token,
    but N tokens ahead via autoregressive rollout. At each step i:
      1. Student predicts logits at current position
      2. Compare with teacher's logits shifted by i positions
      3. Compute KD loss with exponential decay weight (decay^i)
      4. Shift inputs for next step using teacher's argmax prediction
    
    This directly trains the draft model for the speculative decoding task
    where it must predict multiple consecutive tokens accurately.
    """
    
    def __init__(
        self,
        teacher_model,
        l2l_loss_scale,
        train_time_test_steps=7,
        train_time_test_decay=0.8,
        train_time_test_loss_scale=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(teacher_model, l2l_loss_scale, *args, **kwargs)
        self.ttt_steps = train_time_test_steps
        self.ttt_decay = train_time_test_decay
        self.ttt_loss_scale = train_time_test_loss_scale
    
    @staticmethod
    def _shift_left(tensor):
        """Shift tensor left by 1 along seq dim: [a,b,c,d] -> [b,c,d,0]."""
        if tensor.dim() == 2:
            # (batch, seq) — e.g. input_ids
            return torch.cat([tensor[:, 1:], torch.zeros_like(tensor[:, :1])], dim=1)
        elif tensor.dim() == 3:
            # (batch, seq, hidden) — e.g. logits
            return torch.cat([tensor[:, 1:], torch.zeros_like(tensor[:, :1])], dim=1)
        else:
            return tensor
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["output_hidden_states"] = True
        
        # ============================================================
        # Step 0: Teacher forward (single pass, cached for all rollout steps)
        # ============================================================
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        teacher_logits = teacher_outputs.get("logits")  # (B, S, V)
        teacher_reps = teacher_outputs.hidden_states[1:]
        del teacher_outputs
        
        # ============================================================
        # Step 0: Student forward (standard single-step KD)
        # ============================================================
        outputs = model(**inputs)
        
        student_logits = outputs.get("logits")  # (B, S, V)
        student_reps = outputs.hidden_states[1:]
        
        # Standard KD loss (step 0)
        kd_loss_0 = self.ce_loss(student_logits, teacher_logits)
        
        # L2L loss (only at step 0)
        l2l_loss = 0
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            l2l_loss += self.mse_loss(student_rep, teacher_rep)
        l2l_loss = self.l2l_loss_scale * l2l_loss
        
        # Per-position accuracy for step 0
        with torch.no_grad():
            teacher_argmax = teacher_logits.argmax(dim=-1)  # (B, S)
            student_argmax = student_logits.argmax(dim=-1)
            # Use attention_mask to skip padding
            attn_mask = inputs.get("attention_mask", torch.ones_like(teacher_argmax))
            valid_tokens = attn_mask.sum().item()
            acc_0 = ((student_argmax == teacher_argmax) * attn_mask).sum().item() / max(valid_tokens, 1)
        
        if not return_outputs:
            del outputs
        
        # ============================================================
        # Multi-step rollout (training-time test)
        # ============================================================
        rollout_losses = [kd_loss_0]
        rollout_accs = [acc_0]
        
        # Shift teacher targets for future positions
        # At step i, we want teacher_logits shifted left by i positions
        shifted_teacher_logits = teacher_logits
        shifted_teacher_argmax = teacher_argmax
        shifted_mask = attn_mask
        
        # Current student input_ids for rollout
        # Use teacher's argmax as next input (teacher-guided rollout)
        current_input_ids = inputs["input_ids"].clone()
        
        for step_i in range(1, self.ttt_steps):
            # Shift teacher targets left by 1 (predicting one position further)
            shifted_teacher_logits = self._shift_left(shifted_teacher_logits)
            shifted_teacher_argmax = self._shift_left(shifted_teacher_argmax)
            shifted_mask = self._shift_left(shifted_mask)
            
            # Shift student inputs left too (simulating autoregressive advance)
            current_input_ids = self._shift_left(current_input_ids)
            
            # Create new inputs for this step
            step_inputs = {
                "input_ids": current_input_ids,
                "attention_mask": inputs.get("attention_mask", None),
            }
            if step_inputs["attention_mask"] is None:
                step_inputs["attention_mask"] = torch.ones_like(current_input_ids)
            
            # Student forward at this rollout step
            step_outputs = model(**step_inputs)
            step_student_logits = step_outputs.get("logits")
            del step_outputs
            
            # KD loss at this shifted position
            step_kd_loss = self.ce_loss(step_student_logits, shifted_teacher_logits)
            rollout_losses.append(step_kd_loss)
            
            # Accuracy at this step
            with torch.no_grad():
                step_student_argmax = step_student_logits.argmax(dim=-1)
                step_valid = shifted_mask.sum().item()
                step_acc = ((step_student_argmax == shifted_teacher_argmax) * shifted_mask).sum().item() / max(step_valid, 1)
                rollout_accs.append(step_acc)
        
        # ============================================================
        # Aggregate weighted rollout loss
        # ============================================================
        ttt_loss = 0
        for i, rl in enumerate(rollout_losses):
            weight = self.ttt_decay ** i
            ttt_loss += weight * rl
        ttt_loss = self.ttt_loss_scale * ttt_loss
        
        total_loss = ttt_loss + l2l_loss
        
        # Log all metrics
        log_dict = {
            "l2l_loss": l2l_loss.item() if isinstance(l2l_loss, torch.Tensor) else l2l_loss,
            "kd_loss_step0": kd_loss_0.item(),
            "ttt_loss": ttt_loss.item(),
            "total_loss": total_loss.item(),
        }
        for i, acc in enumerate(rollout_accs):
            log_dict[f"acc_{i}"] = acc
        
        self.log(log_dict)
        
        return (total_loss, outputs) if return_outputs else total_loss
