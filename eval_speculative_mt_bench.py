"""
MT-Bench Speculative Decoding Evaluation

Runs serial speculative decoding on the 80 MT-Bench questions (Turn 1)
and measures TPS, acceptance rate, and acceptance length.

Supports:
  - Draft on CPU (C++ LittleBit kernel) or GPU (PyTorch quantized)
  - Target as Matryoshka (0.1+0.9 bit) or FP baseline

No LLM judge — purely measures decoding speed and quality metrics.

Usage:
    python eval_speculative_mt_bench.py \
        --base_model_id /path/to/Llama-3.1-8B-Instruct \
        --draft_model_path outputs/step1_draft_0.1bit/<ts>_runtime \
        --residual_model_path outputs/step2_residual_0.9bit/<ts> \
        --target_mode matryoshka \
        --draft_device cpu_kernel \
        --max_questions 80
"""

import argparse
import json
import os
import time
from typing import List, Dict

import torch

from speculative_decoding import (
    speculative_decode, autoregressive_generate,
    load_target_model, load_draft_model, str2bool,
)
from utils.datautils import load_tokenizer
from utils.misc import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# MT-Bench Turn-1 Questions (all 80)
# ==============================================================================

MT_BENCH_QUESTIONS = [
    # Writing
    {"question_id": 81, "category": "writing", "prompt": "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."},
    {"question_id": 82, "category": "writing", "prompt": "Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask for their review, suggestions, and deadline for revisions."},
    {"question_id": 83, "category": "writing", "prompt": "Describe a vivid and unique character, including their appearance, personality, background, and motivations."},
    {"question_id": 84, "category": "writing", "prompt": "Write a persuasive essay to convince a friend to start a regular exercise routine."},
    {"question_id": 85, "category": "writing", "prompt": "Describe a vivid, detailed dream you've had. Include the setting, characters, events, and any emotions you experienced during the dream."},
    {"question_id": 86, "category": "writing", "prompt": "Write a descriptive paragraph about a bustling marketplace, incorporating sensory details such as sights, sounds, smells, and textures."},
    {"question_id": 87, "category": "writing", "prompt": "Could you write a captivating short story beginning with the sentence: The old abandoned house at the end of the street held a secret that no one had ever discovered."},
    {"question_id": 88, "category": "writing", "prompt": "Craft an intriguing opening paragraph for a fictional short story. The story should involve a character who wakes up one morning to find that they can see everyone's emotions as colorful auras."},
    {"question_id": 89, "category": "writing", "prompt": "Help me construct a catchy, yet scientifically accurate, parsing of the information in this claim: 'Every year, the human body eliminates approximately 60,000 miles of blood vessels as a result of losing fat cells.'"},
    {"question_id": 90, "category": "writing", "prompt": "Edit the following paragraph to correct any grammatical errors:\nShe didn't remembre where is the library was. She, tried to ask a passerby, but they didn't hear her. The library was further than she think."},
    # Roleplay
    {"question_id": 91, "category": "roleplay", "prompt": "Pretend yourself to be Elon Musk in all the following conversations. Speak like Elon Musk as much as possible. Why do we need to go to Mars?"},
    {"question_id": 92, "category": "roleplay", "prompt": "Embrace the role of Sheldon from \"The Big Bang Theory\" as we delve into our conversation. Don't break character. Begin by stating your catchphrase."},
    {"question_id": 93, "category": "roleplay", "prompt": "Imagine yourself as a doctor explaining a complex medical condition to a patient. Use simple language and analogies to help the patient understand."},
    {"question_id": 94, "category": "roleplay", "prompt": "Please take on the role of a relationship counselor. You'll be helping a couple work through their issues and improve their communication. In this scenario, one partner is named Alex and the other is named Jordan."},
    {"question_id": 95, "category": "roleplay", "prompt": "Please assume the role of an English language teacher. I will provide you with a sentence, and you should check it for any grammatical or spelling errors and provide corrections. If the sentence is correct, please confirm it. Here is the sentence: 'He go to school everyday.'"},
    {"question_id": 96, "category": "roleplay", "prompt": "Now you are a machine learning engineer at a tech company. You need to explain to a non-technical manager why the current AI model needs to be updated."},
    {"question_id": 97, "category": "roleplay", "prompt": "Act as a math tutor. I will provide mathematical equations or concepts, and you will explain them in simple terms. Provide step-by-step instructions with examples. My first question is, 'What is algebra?'"},
    {"question_id": 98, "category": "roleplay", "prompt": "Embody the persona of Tony Stark from the Marvel Cinematic Universe. Maintain his characteristic wit, sarcasm, and confidence in your responses. Start by introducing yourself in Stark's style."},
    {"question_id": 99, "category": "roleplay", "prompt": "Suppose you are a mathematician and philosopher. You are going to explain the significance of Gödel's incompleteness theorems to a general audience. Please start by summarizing the theorems in layman's terms."},
    {"question_id": 100, "category": "roleplay", "prompt": "Pretend to be a world-class chef. Give me a recipe for a delicious chocolate cake. Explain each step as if I'm a beginner in cooking."},
    # Reasoning
    {"question_id": 101, "category": "reasoning", "prompt": "Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?"},
    {"question_id": 102, "category": "reasoning", "prompt": "You need to solve a murder mystery. Here are the clues: 1. The victim was found in the study. 2. There were muddy footprints leading to the study. 3. The murder weapon was a candlestick found in the library. 4. A broken window was found in the kitchen. Who is the most likely suspect?"},
    {"question_id": 103, "category": "reasoning", "prompt": "Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?"},
    {"question_id": 104, "category": "reasoning", "prompt": "David has three sisters. Each of them has one brother. How many brothers does David have?"},
    {"question_id": 105, "category": "reasoning", "prompt": "Read the below passage carefully and answer the questions with an explanation:\nAt a small company, parking spaces are reserved for the top executives. The weights of their vehicles are as follows: 3000 lbs, 3200 lbs, 3500 lbs, 3800 lbs, 4000 lbs, 4200 lbs.\n\nQuestion: If the digit of the weight of the lightest car and the heaviest car were swapped, what would be the new total weight of the lightest and heaviest car combined?"},
    {"question_id": 106, "category": "reasoning", "prompt": "Each problem consists of three statements. Based on the first two statements, determine whether the third statement is true, false, or uncertain.\n1. Oranges cost more than apples.\n2. Oranges cost less than bananas.\n3. Bananas cost more than apples and bananas cost more than oranges.\nAnswer: "},
    {"question_id": 107, "category": "reasoning", "prompt": "A is the father of B. B is the father of C. What is the relationship between A and C?"},
    {"question_id": 108, "category": "reasoning", "prompt": "Which word does not belong with the others?\ntyre, steering wheel, engine, trunk, bonnet"},
    {"question_id": 109, "category": "reasoning", "prompt": "One morning after sunrise, Suresh was standing facing a pole. The shadow of the pole fell exactly to his right. To which direction was he facing?"},
    {"question_id": 110, "category": "reasoning", "prompt": "Parents have complained to the principal about bullying during recess. The principal wants to end the bullying and is considering two options:\nOption 1: The principal proposes a round-table discussion with the bullies, their parents and the victims.\nOption 2: The principal proposes to increase recess supervision by hiring more monitors.\nDiscuss the pros and cons of each option, considering the different stakeholders involved."},
    # Math
    {"question_id": 111, "category": "math", "prompt": "The vertices of a triangle are at points (0, 0), (1, 0), and (0.5, 0.866). What is the area of the triangle?"},
    {"question_id": 112, "category": "math", "prompt": "A tech startup invests $8000 in developing a new app. The app generates a monthly revenue of $500 for the first 5 months, and then $2000 per month for the next 5 months. How long does it take for the startup to break even?"},
    {"question_id": 113, "category": "math", "prompt": "In a survey conducted at a local high school, preferences for a new school color were measured: 58% February chose blue, 45% February chose green, and 22% February chose both blue and green. If we pick a student randomly from the school, what is the probability that they would prefer neither blue nor green?"},
    {"question_id": 114, "category": "math", "prompt": "When rolling two satisfactory dice, what is the probability of getting a sum equal to 9?"},
    {"question_id": 115, "category": "math", "prompt": "Some people got on a bus at the terminal. At the first bus stop, half the people got off and 4 more people got on. Then at the second bus stop, 6 people got off and 8 got on. If there are now 25 people on the bus, how many people got on the bus at the terminal?"},
    {"question_id": 116, "category": "math", "prompt": "x+y = 4z, xz = y^2, express x-y in z."},
    {"question_id": 117, "category": "math", "prompt": "How many integers are in the solution of the inequality |x + 5| < 10?"},
    {"question_id": 118, "category": "math", "prompt": "When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?"},
    {"question_id": 119, "category": "math", "prompt": "Benjamin went to a bookstore and purchased a variety of books. He bought 5 copies of a fantasy novel, each priced at $20, 3 copies of a science fiction novel, each priced at $15, and 2 copies of a philosophy book, each priced at $45. What was the total cost of his purchase?"},
    {"question_id": 120, "category": "math", "prompt": "Given that f(x) = 4x^3 - 9x - 14, find the value of f(2)."},
    # Extraction (coding)
    {"question_id": 121, "category": "extraction", "prompt": "Develop a C++ program that reads a text file line by line and counts the number of occurrences of a specific word in the file."},
    {"question_id": 122, "category": "extraction", "prompt": "Write a C++ program to find the nth Fibonacci number using recursion."},
    {"question_id": 123, "category": "extraction", "prompt": "Write a simple website in HTML. When a user clicks a button, it shows a random joke fetched from an API."},
    {"question_id": 124, "category": "extraction", "prompt": "Here is a Python function to find the length of the longest common subsequence of two input strings. Can you identify any bug in this function?\n\ndef longest_common_subsequence_length(str1, str2):\n    m = len(str1)\n    n = len(str2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if str1[i - 1] == str2[j - 1]:\n                dp[i][j] = dp[i - 1][j - 1] + 1\n            else:\n                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n    return dp[m][n]"},
    {"question_id": 125, "category": "extraction", "prompt": "Write a function to find the highest common ancestor (not lowest) of two nodes in a binary tree."},
    {"question_id": 126, "category": "extraction", "prompt": "Implement a function to find the median of two sorted arrays of different sizes with O(1) space complexity and O(n) time complexity."},
    {"question_id": 127, "category": "extraction", "prompt": "Write a function to find the majority element in a given integer array using the Boyer-Moore Voting Algorithm."},
    {"question_id": 128, "category": "extraction", "prompt": "A binary tree is full if all of its vertices have either zero or two children. Let B_n denote the number of full binary trees with n vertices. Implement a function to find B_n."},
    {"question_id": 129, "category": "extraction", "prompt": "You are given two sorted lists of size m and n. Implement a function to find the kth smallest element in the union of the two lists with linear complexity."},
    {"question_id": 130, "category": "extraction", "prompt": "Implement a program to find the common elements in two arrays without using any extra data structures."},
    # STEM
    {"question_id": 131, "category": "stem", "prompt": "Evaluate the following movie reviews on a scale of 1 to 5, with 1 being very negative, 3 being neutral, and 5 being very positive:\n1. This movie released on Nov. 18, 2019, was phenomenal. The storyline was captivating, the characters were well-developed, and the cinematography was stunning.\n2. I was eagerly anticipating this movie, but it fell flat. The plot was predictable, the acting was mediocre, and the special effects were lackluster.\n3. This movie had a mixed reception. Some praised its ambitious storytelling and visual effects, while others felt it was overly complicated and lacked emotional depth.\nReturn the answer as a JSON array of integers."},
    {"question_id": 132, "category": "stem", "prompt": "Given these categories - Literature, History, Science, and Art. Please analyze the following questions and assign them to the appropriate category. In the format of \"Question 1: xx, Question 2: xx, ...\"\n1. What year did the Mona Lisa painting was created?\n2. Who wrote 'Romeo and Juliet'?\n3. What is the boiling point of water?\n4. Who painted 'Starry Night'?"},
    {"question_id": 133, "category": "stem", "prompt": "Extract the following information from the presented texts: name, position, and company. Output should be in JSON format.\n1. In an interview, John emphasized his role as CEO of TechCorp and his vision for the company.\n2. As the Head of Product at InnovateTech, Jane described how she leads cross-functional teams to launch successful products.\n3. Sam claimed his position as Director of Engineering at FutureSoft."},
    {"question_id": 134, "category": "stem", "prompt": "Given the following data, identify the company with the highest profit in 2021 and provide its CEO.\n\na) Company A: Revenue = $10 million, Costs = $8 million\nb) Company B: Revenue = $8 million, Costs = $6 million\nc) Company C: Revenue = $15 million, Costs = $12 million\nd) Company D: Revenue = $12 million, Costs = $10 million"},
    {"question_id": 135, "category": "stem", "prompt": "Identify the countries, their capitals, and the languages spoken in the following sentences. Output them in JSON format.\na) In Paris, the capital of France, people speak French.\nb) In Tokyo, the capital of Japan, people speak Japanese.\nc) The capital of Australia is Canberra and its people predominantly speak English.\nd) Berlin is the capital of Germany and German is spoken there."},
    {"question_id": 136, "category": "stem", "prompt": "Please read the paragraph below and count the number of times the words \"Amazon\", \"river\", and \"forest\" appear. Please present the results in the format of \"word: count\" in a JSON object.\n\nThe Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela."},
    {"question_id": 137, "category": "stem", "prompt": "Identify the below sentences as either a simile or a metaphor. Explain your reasoning.\n1. He was a raging bull at the meeting.\n2. She fought like a lioness for her children.\n3. After the diagnosis, her world fell apart like a house of cards.\n4. He is the apple of my eye."},
    {"question_id": 138, "category": "stem", "prompt": "Analyze the following customer reviews for a restaurant and classify them as positive, negative, or mixed. Output must be in JSON format.\n1. The food was delicious and the service was excellent, but the wait time was too long.\n2. I was disappointed by the cold food and disinterested staff.\n3. An incredible dining experience from start to finish! Every dish was a masterpiece.\n4. The decor was nice, but the menu was overpriced for the portion sizes."},
    {"question_id": 139, "category": "stem", "prompt": "Given a set of complex equations, extract all unique variable names from each equation. Return the results as a JSON string, with one line per equation.\n1) y = (3/4)x^3 - e^(2x) + sin(pi*x) - sqrt(7)\n2) 2A - B/(3+C) * sum(D^2, E, F) = 53.5\n3) z = (R+3)^5 * log(7K, L, M^2)"},
    {"question_id": 140, "category": "stem", "prompt": "Given the following data, identify the highest and lowest values for each column and calculate the average for each column. Format your response as a JSON object.\na) Temperature (°C): [25, 30, 22, 28, 35, 20, 18, 32, 27, 24]\nb) Humidity (%): [50, 55, 60, 45, 70, 40, 65, 58, 42, 48]\nc) Wind Speed (km/h): [15, 20, 10, 25, 30, 12, 18, 22, 8, 14]"},
    # Humanities
    {"question_id": 141, "category": "humanities", "prompt": "In the context of a fictional world, describe a detailed plan for a city that is entirely powered by renewable energy. Consider aspects like infrastructure, transportation, and citizen lifestyle."},
    {"question_id": 142, "category": "humanities", "prompt": "Please consider the ethical implications of using AI in the hiring process and provide an analysis."},
    {"question_id": 143, "category": "humanities", "prompt": "Photosynthesis is a vital process for life on Earth. Could you outline the two main stages of photosynthesis, including where they take place within the chloroplast, and the primary inputs and outputs for each stage?"},
    {"question_id": 144, "category": "humanities", "prompt": "What is the central dogma of molecular biology? Describe the flow of genetic information within a biological system."},
    {"question_id": 145, "category": "humanities", "prompt": "Describe the process and benefits of adopting a zero-waste lifestyle. What are the key principles, and how can individuals effectively transition to this lifestyle?"},
    {"question_id": 146, "category": "humanities", "prompt": "Please explain the concept of dark matter and its significance in our understanding of the universe."},
    {"question_id": 147, "category": "humanities", "prompt": "The city of Veridale is planning to implement a new traffic management system. As an urban planner, what factors would you consider in designing this system? Include details about technology, community impact, and sustainability."},
    {"question_id": 148, "category": "humanities", "prompt": "You have been tasked with designing a solar-powered water heating system for a residential building. Describe the key components and considerations you would include in your design. Draw upon principles from engineering and physics."},
    {"question_id": 149, "category": "humanities", "prompt": "Please describe the concept of Maslow's Hierarchy of Needs and explain how it can be applied to workplace motivation and employee satisfaction."},
    {"question_id": 150, "category": "humanities", "prompt": "How have the Alps influenced the cultural, economic, and political development of the regions surrounding them throughout history?"},
    # Coding
    {"question_id": 151, "category": "coding", "prompt": "Provide a Python function that takes a list and returns the k-th largest element using a min-heap. The function should handle edge cases, use only the standard library, and include docstrings and comments."},
    {"question_id": 152, "category": "coding", "prompt": "How do you reverse a string in Python?"},
    {"question_id": 153, "category": "coding", "prompt": "Write a Python program that reads a file and counts the occurrences of each word in the file. The program should output the top 5 most common words and their counts."},
    {"question_id": 154, "category": "coding", "prompt": "Create a Python function that generates a list of prime numbers up to a given number n."},
    {"question_id": 155, "category": "coding", "prompt": "Write a Python function to find the longest common prefix among a list of strings."},
    {"question_id": 156, "category": "coding", "prompt": "Write a Python function to check if a given string is a valid IPv4 address."},
    {"question_id": 157, "category": "coding", "prompt": "Write a Python function that merges two sorted lists into a single sorted list without using any built-in sort functions."},
    {"question_id": 158, "category": "coding", "prompt": "Write a Python function that sorts a list of dictionaries based on a specified key."},
    {"question_id": 159, "category": "coding", "prompt": "Write a Python function that converts a given string to Pig Latin."},
    {"question_id": 160, "category": "coding", "prompt": "Implement a Python class for a binary search tree with methods for insertion, search, and in-order traversal."},
]


def run_eval(args):
    """Run speculative decoding evaluation on MT-Bench questions."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    greedy = args.mode == "greedy"

    # Select questions
    questions = MT_BENCH_QUESTIONS[:args.max_questions]
    logger.info(f"Evaluating {len(questions)} MT-Bench questions")
    logger.info(f"  Draft device: {args.draft_device}")
    logger.info(f"  Target mode:  {args.target_mode}")
    logger.info(f"  Decode mode:  serial (K={args.draft_length})")
    logger.info(f"  Max tokens:   {args.max_new_tokens}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.base_model_id)
    eos_token_id = tokenizer.eos_token_id or 2
    # Add stop tokens
    stop_token_ids = {eos_token_id}
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        for stop_tok in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
            tid = tokenizer.convert_tokens_to_ids(stop_tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                stop_token_ids.add(tid)

    # Load models
    logger.info("Loading draft model...")
    draft_model = load_draft_model(args, device)

    logger.info("Loading target model...")
    target_model = load_target_model(args, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # =====================================================
    # Phase 1: Autoregressive Baseline (target only)
    # =====================================================
    if args.run_baseline:
        logger.info("=" * 70)
        logger.info("Phase 1: Autoregressive Baseline (target model only)")
        logger.info("=" * 70)

        baseline_results = []
        total_baseline_tokens = 0
        total_baseline_time = 0.0

        for idx, q in enumerate(questions):
            qid = q["question_id"]
            cat = q["category"]
            prompt = q["prompt"]

            chat_messages = [{"role": "user", "content": prompt}]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    chat_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = f"user: {prompt}\nassistant: "

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            output_ids, stats = autoregressive_generate(
                target_model=target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                greedy=greedy,
                eos_token_id=eos_token_id,
            )

            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            total_baseline_tokens += stats["total_tokens_generated"]
            total_baseline_time += stats["elapsed_seconds"]

            result = {
                "question_id": qid,
                "category": cat,
                "tokens_generated": stats["total_tokens_generated"],
                "elapsed_seconds": stats["elapsed_seconds"],
                "tokens_per_second": stats["tokens_per_second"],
                "response_preview": response[:200],
            }
            baseline_results.append(result)

            logger.info(
                f"  [Baseline] Q{qid} ({cat}) [{idx+1}/{len(questions)}] "
                f"{stats['total_tokens_generated']} tokens, "
                f"{stats['tokens_per_second']:.1f} TPS, "
                f"{stats['elapsed_seconds']:.1f}s"
            )

        baseline_avg_tps = total_baseline_tokens / max(total_baseline_time, 1e-6)

        logger.info(f"\n  Baseline Summary:")
        logger.info(f"    Total tokens: {total_baseline_tokens}")
        logger.info(f"    Total time:   {total_baseline_time:.1f}s")
        logger.info(f"    Overall TPS:  {baseline_avg_tps:.2f}")
    else:
        baseline_avg_tps = 0
        baseline_results = []
        total_baseline_tokens = 0
        total_baseline_time = 0.0

    # =====================================================
    # Phase 2: Speculative Decoding
    # =====================================================
    logger.info("=" * 70)
    logger.info(f"Phase 2: Speculative Decoding (K={args.draft_length})")
    logger.info("=" * 70)

    spec_results = []
    total_spec_tokens = 0
    total_spec_time = 0.0
    total_accepted = 0
    total_drafted = 0
    total_steps = 0

    for idx, q in enumerate(questions):
        qid = q["question_id"]
        cat = q["category"]
        prompt = q["prompt"]

        chat_messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = f"user: {prompt}\nassistant: "

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # Reset CPU draft model cache for each new prompt
        if hasattr(draft_model, 'reset'):
            draft_model.reset()

        try:
            output_ids, stats = speculative_decode(
                draft_model=draft_model,
                target_model=target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                draft_length=args.draft_length,
                temperature=1.0 if not greedy else 1.0,
                greedy=greedy,
                eos_token_id=eos_token_id,
                verbose=False,
            )

            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            total_spec_tokens += stats["total_tokens_generated"]
            total_spec_time += stats["elapsed_seconds"]
            total_accepted += stats["total_accepted_tokens"]
            total_drafted += stats["total_draft_tokens"]
            total_steps += stats["num_steps"]

            result = {
                "question_id": qid,
                "category": cat,
                "tokens_generated": stats["total_tokens_generated"],
                "elapsed_seconds": stats["elapsed_seconds"],
                "tokens_per_second": stats["tokens_per_second"],
                "acceptance_rate": stats["acceptance_rate"],
                "mean_acceptance_length": stats["mean_acceptance_length"],
                "num_steps": stats["num_steps"],
                "response_preview": response[:200],
            }
            spec_results.append(result)

            logger.info(
                f"  [SpecDec] Q{qid} ({cat}) [{idx+1}/{len(questions)}] "
                f"{stats['total_tokens_generated']} tokens, "
                f"{stats['tokens_per_second']:.1f} TPS, "
                f"α={stats['mean_acceptance_length']:.2f}, "
                f"accept={stats['acceptance_rate']:.1%}, "
                f"{stats['elapsed_seconds']:.1f}s"
            )
            # Show response preview
            logger.info(f"    [A] {response[:300]}{'...' if len(response) > 300 else ''}")

        except Exception as e:
            logger.error(f"  [SpecDec] Q{qid} FAILED: {e}")
            import traceback
            traceback.print_exc()
            spec_results.append({
                "question_id": qid, "category": cat,
                "tokens_generated": 0, "error": str(e),
            })

    # =====================================================
    # Summary
    # =====================================================
    spec_avg_tps = total_spec_tokens / max(total_spec_time, 1e-6)
    global_acceptance_rate = total_accepted / max(total_drafted, 1)
    global_acceptance_length = total_accepted / max(total_steps, 1)

    print("\n" + "=" * 80)
    print("SPECULATIVE DECODING MT-BENCH RESULTS")
    print("=" * 80)
    print(f"  Draft:  {args.draft_device} | Target: {args.target_mode}")
    print(f"  K={args.draft_length} | Questions: {len(questions)} | Max tokens: {args.max_new_tokens}")
    print("-" * 80)

    if args.run_baseline:
        print(f"\n  BASELINE (Autoregressive target-only):")
        print(f"    Total tokens: {total_baseline_tokens}")
        print(f"    Total time:   {total_baseline_time:.1f}s")
        print(f"    Overall TPS:  {baseline_avg_tps:.2f}")

    print(f"\n  SPECULATIVE DECODING (K={args.draft_length}):")
    print(f"    Total tokens: {total_spec_tokens}")
    print(f"    Total time:   {total_spec_time:.1f}s")
    print(f"    Overall TPS:  {spec_avg_tps:.2f}")
    print(f"    Acceptance rate:   {global_acceptance_rate:.1%}")
    print(f"    Mean accept len:   {global_acceptance_length:.3f}")
    print(f"    Total steps:       {total_steps}")

    if args.run_baseline and baseline_avg_tps > 0:
        speedup = spec_avg_tps / baseline_avg_tps
        print(f"\n  SPEEDUP: {speedup:.2f}x")

    # Per-category breakdown
    categories = ["writing", "roleplay", "reasoning", "math", "extraction", "stem", "humanities", "coding"]
    print(f"\n  {'Category':<12s} | {'#Q':>3s} | {'TPS':>7s} | {'Accept%':>8s} | {'α':>6s}")
    print(f"  {'-'*12}-+-{'-'*3}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")
    for cat in categories:
        cat_results = [r for r in spec_results if r.get("category") == cat and r.get("tokens_generated", 0) > 0]
        if cat_results:
            cat_tps = sum(r["tokens_per_second"] for r in cat_results) / len(cat_results)
            cat_acc = sum(r.get("acceptance_rate", 0) for r in cat_results) / len(cat_results)
            cat_alpha = sum(r.get("mean_acceptance_length", 0) for r in cat_results) / len(cat_results)
            print(f"  {cat:<12s} | {len(cat_results):>3d} | {cat_tps:>7.2f} | {cat_acc:>7.1%} | {cat_alpha:>6.3f}")
    print("=" * 80)

    # Save results
    output = {
        "config": {
            "base_model_id": args.base_model_id,
            "draft_model_path": args.draft_model_path,
            "residual_model_path": args.residual_model_path,
            "target_mode": args.target_mode,
            "draft_device": args.draft_device,
            "draft_length": args.draft_length,
            "max_new_tokens": args.max_new_tokens,
            "max_questions": args.max_questions,
            "mode": args.mode,
        },
        "baseline": {
            "total_tokens": total_baseline_tokens,
            "total_time": total_baseline_time,
            "overall_tps": baseline_avg_tps,
            "per_question": baseline_results,
        },
        "speculative": {
            "total_tokens": total_spec_tokens,
            "total_time": total_spec_time,
            "overall_tps": spec_avg_tps,
            "global_acceptance_rate": global_acceptance_rate,
            "global_acceptance_length": global_acceptance_length,
            "total_steps": total_steps,
            "speedup": spec_avg_tps / max(baseline_avg_tps, 1e-6) if args.run_baseline else "N/A",
            "per_question": spec_results,
        },
    }

    output_path = os.path.join(args.output_dir, "speculative_mt_bench_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MT-Bench Speculative Decoding Evaluation")
    parser.add_argument("--base_model_id", type=str, required=True,
                        help="Base model ID (for tokenizer and FP target)")
    parser.add_argument("--draft_model_path", type=str, required=True,
                        help="Path to draft model (HF checkpoint or runtime dir)")
    parser.add_argument("--residual_model_path", type=str, default=None,
                        help="Path to 0.9-bit residual model (for target_mode=matryoshka)")
    parser.add_argument("--target_mode", type=str, default="matryoshka",
                        choices=["fp", "matryoshka"],
                        help="Target model: fp=original FP, matryoshka=0.1+0.9 combined")
    parser.add_argument("--draft_device", type=str, default="cpu_kernel",
                        choices=["cuda", "cpu_kernel"],
                        help="Draft model device")
    parser.add_argument("--decode_mode", type=str, default="serial")
    parser.add_argument("--draft_length", type=int, default=5,
                        help="Number of draft tokens per step (K)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_questions", type=int, default=80)
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--run_baseline", type=str2bool, default=True,
                        help="Also run autoregressive baseline for speedup comparison")
    parser.add_argument("--output_dir", type=str, default="eval_results/speculative_mt_bench")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
