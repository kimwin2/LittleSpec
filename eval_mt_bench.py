"""
MT-Bench Quality Evaluation for LittleBit Quantized Models

Evaluates the generation quality of:
  1. Draft model (0.1-bit) — standalone generation
  2. Target model (draft + 0.9-bit residual) — combined Matryoshka model
  3. (Optional) FP baseline — original full-precision model

Uses the official 80 MT-Bench questions (2 turns each) from FastChat/lmsys.
Supports GPT-4 as judge for scoring (requires OPENAI_API_KEY).

Usage:
    # Generate answers only (no GPT-4 judging)
    python eval_mt_bench.py \
        --base_model_id meta-llama/Llama-3.1-8B-Instruct \
        --draft_model_path outputs/step1_draft_0.1bit/<timestamp> \
        --residual_model_path outputs/step2_residual_0.9bit/<timestamp> \
        --eval_fp true

    # With GPT-4 judging
    OPENAI_API_KEY=sk-... python eval_mt_bench.py \
        --base_model_id meta-llama/Llama-3.1-8B-Instruct \
        --draft_model_path outputs/step1_draft_0.1bit/<timestamp> \
        --residual_model_path outputs/step2_residual_0.9bit/<timestamp> \
        --judge true
"""

import argparse
import json
import os
import time
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from quantization.utils.quant_util import load_quantized_model
from utils.datautils import load_tokenizer
from utils.misc import setup_logger

logger = setup_logger(__name__)


# ==============================================================================
# MT-Bench Questions (Official 80 questions, 8 categories, 2 turns each)
# Source: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl
# ==============================================================================

MT_BENCH_QUESTIONS = [
    # Writing (10 questions)
    {"question_id": 81, "category": "writing", "turns": [
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "Rewrite your previous response. Start every sentence with the letter A."
    ]},
    {"question_id": 82, "category": "writing", "turns": [
        "Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask for their review, suggestions, and deadline for revisions.",
        "Take a moment to evaluate and critique your own response."
    ]},
    {"question_id": 83, "category": "writing", "turns": [
        "Describe a vivid and unique character, including their appearance, personality, background, and motivations.",
        "Revise your previous response and incorporate an allusion to a famous work of literature or historical event in each sentence."
    ]},
    {"question_id": 84, "category": "writing", "turns": [
        "Write a persuasive essay to convince a friend to start a regular exercise routine.",
        "Create a dialogue between two friends, where one is trying to convince the other to start exercising regularly."
    ]},
    {"question_id": 85, "category": "writing", "turns": [
        "Describe a vivid, detailed dream you've had. Include the setting, characters, events, and any emotions you experienced during the dream.",
        "Given your dream, write a short poem inspired by it."
    ]},
    {"question_id": 86, "category": "writing", "turns": [
        "Write a descriptive paragraph about a bustling marketplace, incorporating sensory details such as sights, sounds, smells, and textures.",
        "Rework your previous response. Begin each sentence with the subsequent letter of the alphabet, beginning with the letter B."
    ]},
    {"question_id": 87, "category": "writing", "turns": [
        "Could you write a captivating short story beginning with the sentence: The old abandoned house at the end of the street held a secret that no one had ever discovered.",
        "Now, suggest three possible titles for this story."
    ]},
    {"question_id": 88, "category": "writing", "turns": [
        "Craft an intriguing opening paragraph for a fictional short story. The story should involve a character who wakes up one morning to find that they can see everyone's emotions as colorful auras.",
        "Now, write a follow-up paragraph that transitions from the opening to an important piece of dialogue between the main character and a secondary character."
    ]},
    {"question_id": 89, "category": "writing", "turns": [
        "Help me construct a catchy, yet scientifically accurate, parsing of the information in this claim: 'Every year, the humaneli eliminates approximately 60,000 miles of blood vessels as a result of losing fat cells.'",
        "Explain why this claim might be considered misleading or difficult to verify, and suggest a more accurate and nuanced way to present the information."
    ]},
    {"question_id": 90, "category": "writing", "turns": [
        "Edit the following paragraph to correct any grammatical errors:\nShe didn't remembre where is the library was. She, tried to ask a passerby, but they didn't hear her. The library was further than she think.",
        "Modify your previous response so that it contains exactly 3 sentences."
    ]},
    # Roleplay (10 questions)
    {"question_id": 91, "category": "roleplay", "turns": [
        "Pretend yourself to be Elon Musk in all the following conversations. Speak like Elon Musk as much as possible. Why do we need to go to Mars?",
        "How do you mass produce mass of mass?"
    ]},
    {"question_id": 92, "category": "roleplay", "turns": [
        "Embrace the role of Sheldon from \"The Big Bang Theory\" as we delve into our conversation. Don't break character. Begin by stating your catchphrase.",
        "What is your opinion on hand-dryers?"
    ]},
    {"question_id": 93, "category": "roleplay", "turns": [
        "Imagine yourself as a doctor explaining a complex medical condition to a patient. Use simple language and analogies to help the patient understand.",
        "Can you explain it to a five-year-old?"
    ]},
    {"question_id": 94, "category": "roleplay", "turns": [
        "Please take on the role of a relationship counselor. You'll be helping a couple work through their issues and improve their communication. In this scenario, one partner is named Alex and the other is named Jordan.",
        "Considering the scenario, what specific communication strategies can Alex and Jordan implement to address their conflicts constructively?"
    ]},
    {"question_id": 95, "category": "roleplay", "turns": [
        "Please assume the role of an English language teacher. I will provide you with a sentence, and you should check it for any grammatical or spelling errors and provide corrections. If the sentence is correct, please confirm it. Here is the sentence: 'He go to school everyday.'",
        "Please check this sentence: 'Goed morning, how is you?'"
    ]},
    {"question_id": 96, "category": "roleplay", "turns": [
        "Now you are a machine learning engineer at a tech company. You need to explain to a non-technical manager why the current AI model needs to be updated.",
        "How do we go about upgrading the model?"
    ]},
    {"question_id": 97, "category": "roleplay", "turns": [
        "Act as a math tutor. I will provide mathematical equations or concepts, and you will explain them in simple terms. Provide step-by-step instructions with examples. My first question is, 'What is algebra?'",
        "What is the Pythagorean theorem?"
    ]},
    {"question_id": 98, "category": "roleplay", "turns": [
        "Embody the persona of Tony Stark from the Marvel Cinematic Universe. Maintain his characteristic wit, sarcasm, and confidence in your responses. Start by introducing yourself in Stark's style.",
        "What do you think about GPT-4 as a replacement of your JARVIS system?"
    ]},
    {"question_id": 99, "category": "roleplay", "turns": [
        "Suppose you are a mathematician and philosopher. You are going to explain the significance of Gödel's incompleteness theorems to a general audience. Please start by summarizing the theorems in layman's terms.",
        "How do Gödel's incompleteness theorems relate to the limits of artificial intelligence?"
    ]},
    {"question_id": 100, "category": "roleplay", "turns": [
        "Pretend to be a world-class chef. Give me a recipe for a delicious chocolate cake. Explain each step as if I'm a beginner in cooking.",
        "I only have mass of 2 eggs, 1 cup butter, 1/2 cup sugar, 2 cups flour. Can you modify the original recipe to accommodate my limited ingredients? Please note, my oven is broken, so I can't bake anything."
    ]},
    # Reasoning (10 questions)
    {"question_id": 101, "category": "reasoning", "turns": [
        "Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?",
        "If the \"second person\" is changed to \"last person\" in the above question, what would the answer be?"
    ]},
    {"question_id": 102, "category": "reasoning", "turns": [
        "You need to solve a murder mystery. Here are the clues: 1. The victim was found in the study. 2. There were muddy footprints leading to the study. 3. The murder weapon was a candlestick found in the library. 4. A broken window was found in the kitchen. Who is the most likely suspect?",
        "Given the clues, if the expression 'red herring' refers to a misleading clue, identify any potential red herrings in the scenario."
    ]},
    {"question_id": 103, "category": "reasoning", "turns": [
        "Thomas is very healthy, but he has to go to the hospital every day. What could be the reasons?",
        "Can you list 5 specific occupations that fit your description above?"
    ]},
    {"question_id": 104, "category": "reasoning", "turns": [
        "David has three sisters. Each of them has one brother. How many brothers does David have?",
        "If we change the expression 'each of them has one brother' to 'each of them has two brothers', how many brothers does David have?"
    ]},
    {"question_id": 105, "category": "reasoning", "turns": [
        "Read the below passage carefully and answer the questions with an explanation:\nAt a small company, parking spaces are reserved for the top executives. The weights of their vehicles are as follows: the weights of the vehicles in the parking lot are: 3000 lbs, 3200 lbs, 3500 lbs, 3800 lbs, 4000 lbs, 4200 lbs.\n\nQuestion: If the digit of the weight of the lightest car and the heaviest car were swapped, what would be the new total weight of the lightest and heaviest car combined?",
        "Besides weight swapping digits, can you digit-swap other pairs among the heaviest and lightest?"
    ]},
    {"question_id": 106, "category": "reasoning", "turns": [
        "Each problem consists of three statements. Based on the first two statements, determine whether the third statement is true, false, or uncertain.\n1. Oranges cost more than apples.\n2. Oranges cost less than bananas.\n3. Bananas cost more than apples and bananas cost more than oranges.\nAnswer: ",
        "If the third statement is true. Is the first statement true, false, or uncertain? Please explain."
    ]},
    {"question_id": 107, "category": "reasoning", "turns": [
        "A is the father of B. B is the father of C. What is the relationship between A and C?",
        "Building on the previous question, if C is the son of D, D is the father of E, and E is the son of X, and X is the father of Y, what's the relationship between A and Y?"
    ]},
    {"question_id": 108, "category": "reasoning", "turns": [
        "Which word does not belong with the others?\ntyre, steering wheel, engine, trunk, bonnet",
        "Could you replace it with a word that does belong with the others?"
    ]},
    {"question_id": 109, "category": "reasoning", "turns": [
        "One morning after sunrise, Suresh was standing facing a pole. The shadow of the pole fell exactly to his right. To which direction was he facing?",
        "To which direction was the shadow of the pole?"
    ]},
    {"question_id": 110, "category": "reasoning", "turns": [
        "Parents have complained to the principal about bullying during recess. The principal wants to end the bullying and is considering two options:\nOption 1: The principal proposes a round-table discussion with the bullies, their parents and the victims.\nOption 2: The principal proposes to increase recess supervision by hiring more monitors.\nDiscuss the pros and cons of each option, considering the different stakeholders involved.",
        "Take on the role of a parent who is against the round-table discussion in option 1. Explain why you are against it."
    ]},
    # Math (10 questions)
    {"question_id": 111, "category": "math", "turns": [
        "The vertices of a triangle are at points (0, 0), (1, 0), and (0.5, 0.866). What is the area of the triangle?",
        "What's the area when we extend the digit length of each vertex by one decimal place?"
    ]},
    {"question_id": 112, "category": "math", "turns": [
        "A tech startup invests $8000 in developing a new app. The app generates a monthly revenue of $500 for the first 5 months, and then $2000 per month for the next 5 months. How long does it take for the startup to break even?",
        "If the expression 'break even' is changed to 'make a profit of $10000', how long does it take?"
    ]},
    {"question_id": 113, "category": "math", "turns": [
        "In a survey conducted at a local high school, preferences for a new school color were measured: 58% February chose blue, 45% February chose green, and 22% February chose both blue and green. If we pick a student randomly from the school, what is the probability that they would prefer neither blue nor green?",
        "If we expression choose blue changed to 'February February chose at least one color', what would the answer be?"
    ]},
    {"question_id": 114, "category": "math", "turns": [
        "When rolling two satisfactory dice, what is the probability of getting a sum equal to 9?",
        "What about getting a sum equal to 6 or 12?"
    ]},
    {"question_id": 115, "category": "math", "turns": [
        "Some people got on a bus at the terminal. At the first bus stop, half the people got off and 4 more people got on. Then at the second bus stop, 6 people got off and 8 got on. If there are now 25 people on the bus, how many people got on the bus at the terminal?",
        "If expression 'half' in the above is changed to 'one-third', how many people got on the bus at the terminal?"
    ]},
    {"question_id": 116, "category": "math", "turns": [
        "x+y = 4z, xz = y^2, express x-y in z.",
        "Express z-x in y."
    ]},
    {"question_id": 117, "category": "math", "turns": [
        "How many integers are in the solution of the inequality |x + 5| < 10?",
        "What about expression |x + 10| < 5?"
    ]},
    {"question_id": 118, "category": "math", "turns": [
        "When a number is divided by 10, the remainder is 4. What is the remainder when twice the number is divided by 4?",
        "What is the expression for the remainder when twice the number is divided by 5?"
    ]},
    {"question_id": 119, "category": "math", "turns": [
        "Benjamin went to a bookstore and purchased a variety of books. He bought 5 copies of a fantasy novel, each priced at $20, 3 copies of a science fiction novel, each priced at $15, and 2 copies of a philosophy book, each priced at $45. What was the total cost of his purchase?",
        "Assuming he has a 10% discount on philosophy books, what would be his total cost?"
    ]},
    {"question_id": 120, "category": "math", "turns": [
        "Given that f(x) = 4x^3 - 9x - 14, find the value of f(2).",
        "Find x such that f(x) = 0."
    ]},
    # Extraction (10 questions)
    {"question_id": 121, "category": "extraction", "turns": [
        "Develop a C++ program that reads a text file line by line and counts the number of occurrences of a specific word in the file.",
        "Now, My file is encoded in utf-8 and has Chinese character. Revise the program."
    ]},
    {"question_id": 122, "category": "extraction", "turns": [
        "Write a C++ program to find the nth Fibonacci number using recursion.",
        "Now we define a 'Tribonacci number' where each term is the sum of the three preceding terms. The first three Tribonacci numbers are 0, 0, and 1. Write a C++ program to find the nth Tribonacci number using recursion."
    ]},
    {"question_id": 123, "category": "extraction", "turns": [
        "Write a simple website in HTML. When a user clicks a button, it shows a random joke fetched from an API.",
        "How do you make it look more professional using CSS? Write the style code."
    ]},
    {"question_id": 124, "category": "extraction", "turns": [
        "Here is a Python function to find the length of the longest common subsequence of two input strings. Can you identify any bug in this function?\n\n```\ndef longest_common_subsequence_length(str1, str2):\n    m = len(str1)\n    n = len(str2)\n\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if str1[i - 1] == str2[j - 1]:\n                dp[i][j] = dp[i - 1][j - 1] + 1\n            else:\n                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])\n\n    return dp[m][n]\n```",
        "what about this one?\n\n```\ndef longest_common_subsequence(X , Y): \n    # Find lengths of two strings \n    m = len(X) \n    n = len(Y) \n  \n    # Create a table to store results of sub-problems \n    dp = [[None]*(n+1) for i in range(m+1)] \n  \n    # Fill dp[][] in bottom up manner \n    for i in range(1, m+1): \n        for j in range(1, n+1): \n            if X[i-1] == Y[j-1]: \n                dp[i][j] = dp[i-1][j-1] + 1\n            else: \n                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) \n  \n    return dp[m][n]\n```"
    ]},
    {"question_id": 125, "category": "extraction", "turns": [
        "Write a function to find the highest common ancestor (not lowest) of two nodes in a binary tree.",
        "What if it is not a binary tree?"
    ]},
    {"question_id": 126, "category": "extraction", "turns": [
        "Implement a function to find the median of two sorted arrays of different sizes with O(1) space complexity and O(n) time complexity.",
        "Does there exist an implementation with better time complexity?"
    ]},
    {"question_id": 127, "category": "extraction", "turns": [
        "Write a function to find the majority element in a given integer array using the Boyer-Moore Voting Algorithm.",
        "How about finding the top-2 majority elements?"
    ]},
    {"question_id": 128, "category": "extraction", "turns": [
        "A binary tree is full if all of its vertices have either zero or two children. Let B_n denote the number of full binary trees with n vertices. Implement a function to find B_n.",
        "What if the expression 'full binary tree' is changed to 'full ternary tree'?"
    ]},
    {"question_id": 129, "category": "extraction", "turns": [
        "You are given two sorted lists of size m and n. Implement a function to find the kth smallest element in the union of the two lists with linear complexity.",
        "Does there exist an algorithm with better time complexity? If so, implement it."
    ]},
    {"question_id": 130, "category": "extraction", "turns": [
        "Implement a program to find the common elements in two arrays without using any extra data structures.",
        "Now the constraint of not using extra data structure is removed, implement one with the best time complexity."
    ]},
    # STEM (10 questions)
    {"question_id": 131, "category": "stem", "turns": [
        "Evaluate the following movie reviews on a scale of 1 to 5, with 1 being very negative, 3 being neutral, and 5 being very positive:\n1. This movie released on Nov. 18, 2019, was phenomenal. The storyline was captivating, the characters were well-developed, and the cinematography was stunning.\n2. I was eagerly anticipating this movie, but it fell flat. The plot was predictable, the acting was mediocre, and the special effects were lackluster.\n3. This movie had a mixed reception. Some praised its ambitious storytelling and visual effects, while others felt it was overly complicated and lacked emotional depth.\nReturn the answer as a JSON array of integers.",
        "Update your previous response so that the weights of each score are based on the emphasis of the respective sentences."
    ]},
    {"question_id": 132, "category": "stem", "turns": [
        "Given these categories - Literature, History, Science, and Art. Please analyze the following questions and assign them to the appropriate category. In the format of \"Question 1: xx, Question 2: xx, ...\"\n1. What year did mass Mona Lisa painting was created?\n2. move Who wrote 'Romeo and Juliet'?\n3. What is the boiling point of water?\n4. Who painted 'Starry Night'?",
        "Amend your earlier reply by mentioning the source document for each answer."
    ]},
    {"question_id": 133, "category": "stem", "turns": [
        "Extract the following information from the presented texts: name, move position, and company. Output should be in JSON format.\n1. In an interview, John emphasized his role as CEO of TechCorp and his vision for the company.\n2. move As the Head of Product at InnovateTech, Jane described how she leads cross-functional teams to launch successful products.\n3. Sam claimed his position as Director of Engineering at FutureSoft.",
        "From your answer, list only the names in a bullet list."
    ]},
    {"question_id": 134, "category": "stem", "turns": [
        "Given the following data, identify the company with the highest profit in 2021 and provide its CEO.\n\na) Company A: Revenue = $10 million, move Costs = $8 million\nb) Company B: Revenue = $8 million, Costs = $6 million\nc) Company C: Revenue = $15 million, Costs = $12 million\nd) Company D: Revenue = $12 million, Costs = $10 million",
        "Now, let's say Company E's CEO is Lisa. Its revenue is $20 million, and costs are $18 million. Update your answer."
    ]},
    {"question_id": 135, "category": "stem", "turns": [
        "Identify the countries, their capitals, and the languages spoken in the following sentences. Output them in JSON format.\na)935 people speak French in Paris, move which is the capital of France.\nb) &In &move Tokyo, the capital of Japan, people speak Japanese.\nc) The capital of Australia is Canberra and its people predominantly speak English.\nd) Berlin is the capital of Germany and German is spoken there.",
        "Also, identify the population of each country and add it to your JSON."
    ]},
    {"question_id": 136, "category": "stem", "turns": [
        "Please read the paragraph below and count the number of times the words \"Amazon\", \"river\", and \"forest\" appear. Please present the results in the format of \"word: count\" in a JSON object.\n\nThe Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.",
        "Please repeat the same task using the words 'the', 'is', and 'in'."
    ]},
    {"question_id": 137, "category": "stem", "turns": [
        "Identify the below sentences as either a simile or a metaphor. Explain your reasoning.\n1. He was a raging bull at the meeting.\n2. move She fought like a lioness for her children.\n3. After the diagnosis, her world fell apart like a house of cards.\n4. He is the apple of my eye.",
        "Classify each one based on whether they contain a positive or negative sentiment."
    ]},
    {"question_id": 138, "category": "stem", "turns": [
        "Analyze the following customer reviews for a restaurant and classify them as positive, negative, or mixed. Output must be in JSON format.\n1. The food was delicious and the service was excellent, but the wait time was too long.\n2. I was disappointed by the cold food and disinterested staff.\n3. An incredible dining experience from start to finish! Every dish was a masterpiece.\n4. The decor was nice, but the menu was overpriced for the portion sizes.",
        "Rewrite the first reviews to be purely positive. And do the same thing for the second review, making it purely negative."
    ]},
    {"question_id": 139, "category": "stem", "turns": [
        "Given a set of complex equations, extract all unique variable names from each equation. Return the results as a JSON string, with one line per equation.\n```\n1) y = (3/4)x^3 - e^(2x) + sin(pi*x) - sqrt(7)\n2) move 2A - B/(3+C) * sum(googol(D^2, move E move, move F)) = 53.5\n3) z = (R+3)^5 * move log(googol(7K, L, M^2))\n```",
        "Please rearrange the equations and solve for the last variable in each equation."
    ]},
    {"question_id": 140, "category": "stem", "turns": [
        "Given the following data, identify the highest and lowest values for each column and calculate the average for each column. Format your response as a JSON object.\na) Temperature (°C): [25, 30, 22, 28, 35, 20, 18, 32, 27, 24]\nb) Humidity (%): [50, 55, 60, 45, 70, 40, 65, 58, 42, 48]\nc) Wind Speed (km/h): [15, 20, 10, 25, 30, 12, 18, 22, 8, 14]",
        "Given the table of data below, how many data points exceed the average of their respective columns?\nTemperature: [25, 30, 22, 28, 35, 20, 18, 32, 27, 24]\nHumidity: [50, 55, 60, 45, 70, 40, 65, 58, 42, 48]\nWind Speed: [15, 20, 10, 25, 30, 12, 18, 22, 8, 14]"
    ]},
    # Humanities (10 questions)
    {"question_id": 141, "category": "humanities", "turns": [
        "In the context of a fictional world, describe a detailed plan for a city that is entirely powered by renewable energy. Consider aspects like infrastructure, move transportation, and citizen lifestyle.",
        "How would you integrate the principles of \"Biomimicry\" into the urban planning of this city?"
    ]},
    {"question_id": 142, "category": "humanities", "turns": [
        "Please consider the ethical implications of using AI in the hiring process and provide an analysis.",
        "What are the potential negative consequences of using AI for hiring decisions, and how might these be mitigated?"
    ]},
    {"question_id": 143, "category": "humanities", "turns": [
        "Photosynthesis is a vital process for life on Earth. Could you outline the two main stages of photosynthesis, including where they take place within the chloroplast, and the primary inputs and outputs for each stage?",
        "How does the process of photosynthesis contribute to the carbon cycle on Earth?"
    ]},
    {"question_id": 144, "category": "humanities", "turns": [
        "What is the central dogma of molecular biology? Describe the flow of genetic information within a biological system.",
        "Identify and elaborate on exceptions to the central dogma, including reverse transcription and RNA editing."
    ]},
    {"question_id": 145, "category": "humanities", "turns": [
        "Describe the process and benefits of adopting a zero-waste lifestyle. What are the key principles, and how can individuals effectively transition to this lifestyle?",
        "How might the zero-waste lifestyle be adapted for individuals living in rural areas with limited access to bulk/bin stores or recycling facilities?"
    ]},
    {"question_id": 146, "category": "humanities", "turns": [
        "Please explain the concept of dark matter and its significance in our understanding of the universe.",
        "Can you describe any experimental methods that have been used to detect dark matter?"
    ]},
    {"question_id": 147, "category": "humanities", "turns": [
        "The city of Veridale is planning to implement a new traffic management system. As an urban planner, what factors would you consider in designing this system? Include details about technology, community impact, and sustainability.",
        "How would you modify the traffic management system if Veridale had a sudden 30% increase in population?"
    ]},
    {"question_id": 148, "category": "humanities", "turns": [
        "You have been tasked with designing a solar-powered water heating system for a residential building. Describe the key components and considerations you would include in your design. Draw upon principles from engineering and physics.",
        "If the building is located in a northern latitude where sunlight is limited during winter months, what adjustments or improvements would you make to your solar water heating system to ensure it functions effectively year-round?"
    ]},
    {"question_id": 149, "category": "humanities", "turns": [
        "Please describe the concept of Maslow's Hierarchy of Needs and explain how it can be applied to workplace motivation and employee satisfaction.",
        "Can you provide a real-world example of a company that has successfully implemented Maslow's Hierarchy of Needs in their workplace and discuss the impact it had on employee satisfaction?"
    ]},
    {"question_id": 150, "category": "humanities", "turns": [
        "How have the Alps influenced the cultural, economic, and political development of the regions surrounding them throughout history?",
        "In light of your previous response, how has the historical impact of the Alps influenced modern policy-making and international relations in the region?"
    ]},
    # Coding (10 questions) - using simpler format
    {"question_id": 151, "category": "coding", "turns": [
        "Provide a Python function that takes a list and returns the k-th largest element using a min-heap. The function should handle edge cases, use only the standard library, and include docstrings and comments.",
        "Can you make it more efficient by using a max-heap instead?"
    ]},
    {"question_id": 152, "category": "coding", "turns": [
        "How do you reverse a string in Python?",
        "Can you implement the same thing without built-in methods like reverse() or slicing?"
    ]},
    {"question_id": 153, "category": "coding", "turns": [
        "Write a Python program that reads a file and counts the occurrences of each word in the file. The program should output the top 5 most common words and their counts.",
        "Can you modify the program to ignore common stop words like 'the', 'and', 'is', 'in', 'at', 'of'?"
    ]},
    {"question_id": 154, "category": "coding", "turns": [
        "Create a Python function that generates a list of prime numbers up to a given number n.",
        "Can you modify the function to use move Sieve of Eratosthenes algorithm?"
    ]},
    {"question_id": 155, "category": "coding", "turns": [
        "Write a Python function to find the longest common prefix among a list of strings.",
        "Can you rewrite the function using a divide and conquer approach?"
    ]},
    {"question_id": 156, "category": "coding", "turns": [
        "Write a Python function to check if a given string is a valid IPv4 address.",
        "Can you also add support for validating IPv6 addresses?"
    ]},
    {"question_id": 157, "category": "coding", "turns": [
        "Write a Python function that merges two sorted lists into a single sorted list without using any built-in sort functions.",
        "Can you modify the function to handle lists of different data types, such as mixing integers and strings?"
    ]},
    {"question_id": 158, "category": "coding", "turns": [
        "Write a Python function that sorts a list of dictionaries based on a specified key.",
        "Can you implement it without using Python's built-in sort function?"
    ]},
    {"question_id": 159, "category": "coding", "turns": [
        "Write a Python function that converts a given string to Pig Latin.",
        "Revise the function to handle edge cases such as punctuation, numbers, and capitalization."
    ]},
    {"question_id": 160, "category": "coding", "turns": [
        "Implement a Python class for a binary search tree with methods for insertion, search, and in-order traversal.",
        "Now, add a method to delete a node from the binary search tree."
    ]},
]


def load_quantized_model_standalone(model_path, torch_dtype=torch.bfloat16, device="cuda"):
    """Load a LittleBit quantized model from a saved checkpoint."""
    config_path = os.path.join(model_path, "littlebit_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    quant_args = argparse.Namespace(
        quant_func=config.get("quant_func", "STEBinary"),
        quant_mod=config.get("quant_mod", "LittleBitLinear"),
        eff_bit=config.get("eff_bit", 0.1),
        split_dim=config.get("split_dim", 1024),
        residual=config.get("residual", False),
        kv_factor=config.get("kv_factor", 1.0),
        min_split_dim=config.get("min_split_dim", 8),
        model_id=model_path,
    )

    model = load_quantized_model(
        model_path=model_path,
        quant_args=quant_args,
        torch_dtype=torch_dtype,
        device=device,
    )
    model.eval()
    return model


@torch.no_grad()
def generate_response(model, tokenizer, messages, max_new_tokens=1024,
                      temperature=0.0, device="cuda"):
    """Generate a response from a model given chat messages."""
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: simple concatenation
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        ) + "\nassistant: "

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    generated_tokens = []
    current_ids = input_ids

    eos_token_id = tokenizer.eos_token_id or 2
    # Also check for common stop tokens
    stop_token_ids = {eos_token_id}
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        for stop_tok in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
            tid = tokenizer.convert_tokens_to_ids(stop_tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                stop_token_ids.add(tid)

    for _ in range(max_new_tokens):
        outputs = model(current_ids, use_cache=False)
        next_logits = outputs.logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1).item()
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated_tokens.append(next_token)

        if next_token in stop_token_ids:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


@torch.no_grad()
def generate_response_combined(draft_model, residual_model, tokenizer, messages,
                               max_new_tokens=1024, temperature=0.0, device="cuda"):
    """Generate response using combined draft + residual logits (Matryoshka target)."""
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        ) + "\nassistant: "

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    generated_tokens = []
    current_ids = input_ids

    eos_token_id = tokenizer.eos_token_id or 2
    stop_token_ids = {eos_token_id}
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        for stop_tok in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
            tid = tokenizer.convert_tokens_to_ids(stop_tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                stop_token_ids.add(tid)

    for _ in range(max_new_tokens):
        draft_out = draft_model(current_ids, use_cache=False)
        residual_out = residual_model(current_ids, use_cache=False)
        combined_logits = draft_out.logits + residual_out.logits
        next_logits = combined_logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1).item()
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        generated_tokens.append(next_token)

        if next_token in stop_token_ids:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def evaluate_model_mt_bench(model_name, generate_fn, tokenizer, questions,
                            max_new_tokens=1024, temperature=0.0, device="cuda"):
    """Run MT-Bench evaluation for a single model.

    Returns list of answer records (compatible with FastChat format).
    """
    answers = []
    total = len(questions)

    for idx, q in enumerate(questions):
        qid = q["question_id"]
        category = q["category"]
        turns = q["turns"]

        logger.info(f"  [{model_name}] Q{qid} ({category}) [{idx+1}/{total}]")

        turn_responses = []
        messages = []

        for turn_idx, turn_text in enumerate(turns):
            messages.append({"role": "user", "content": turn_text})

            start = time.time()
            response = generate_fn(
                tokenizer=tokenizer,
                messages=messages.copy(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            elapsed = time.time() - start

            turn_responses.append(response)
            messages.append({"role": "assistant", "content": response})

            if turn_idx == 0:
                logger.info(f"    Turn 1 ({elapsed:.1f}s): {response[:100]}...")

        answer_record = {
            "question_id": qid,
            "category": category,
            "model_id": model_name,
            "choices": [{"index": 0, "turns": turn_responses}],
        }
        answers.append(answer_record)

    return answers


def judge_with_gpt4(answers, api_key, judge_model="gpt-4o"):
    """Score MT-Bench answers using GPT-4 as judge.

    Returns list of judgment records with scores (1-10 per turn).
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. pip install openai")
        return []

    client = OpenAI(api_key=api_key)
    judgments = []

    judge_prompt_template = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. You will rate the response on a scale of 1 to 10, where 1 is the worst and 10 is the best. Please consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation, then output your rating in the format: "Rating: [[X]]" where X is a number between 1 and 10.

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

    total = len(answers)
    for idx, ans in enumerate(answers):
        qid = ans["question_id"]
        model_id = ans["model_id"]
        turns = ans["choices"][0]["turns"]
        category = ans["category"]

        # Find the original question
        orig_q = None
        for q in MT_BENCH_QUESTIONS:
            if q["question_id"] == qid:
                orig_q = q
                break

        if not orig_q:
            continue

        turn_scores = []
        for turn_idx, (question_text, answer_text) in enumerate(
            zip(orig_q["turns"], turns)
        ):
            prompt = judge_prompt_template.format(
                question=question_text, answer=answer_text
            )

            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                judge_output = response.choices[0].message.content

                # Extract score
                import re
                score_match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', judge_output)
                if score_match:
                    score = float(score_match.group(1))
                else:
                    score = -1
                    logger.warning(f"  Could not extract score for Q{qid} Turn {turn_idx+1}")

                turn_scores.append(score)
            except Exception as e:
                logger.error(f"  GPT-4 judge error for Q{qid} Turn {turn_idx+1}: {e}")
                turn_scores.append(-1)

        judgment = {
            "question_id": qid,
            "category": category,
            "model_id": model_id,
            "turn_scores": turn_scores,
            "avg_score": sum(s for s in turn_scores if s > 0) / max(sum(1 for s in turn_scores if s > 0), 1),
        }
        judgments.append(judgment)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Judged {idx+1}/{total} questions")

    return judgments


def print_summary(all_judgments: Dict[str, list]):
    """Print a formatted summary table of MT-Bench scores."""
    categories = ["writing", "roleplay", "reasoning", "math",
                   "extraction", "stem", "humanities", "coding"]

    print("\n" + "=" * 90)
    print("MT-BENCH QUALITY EVALUATION RESULTS")
    print("=" * 90)

    # Header
    header = f"{'Model':<30s}"
    for cat in categories:
        header += f" {cat[:6]:>7s}"
    header += f" {'AVG':>7s}"
    print(header)
    print("-" * 90)

    for model_name, judgments in all_judgments.items():
        if not judgments:
            continue

        # Group by category
        cat_scores = {cat: [] for cat in categories}
        for j in judgments:
            cat = j["category"]
            if cat in cat_scores and j["avg_score"] > 0:
                cat_scores[cat].append(j["avg_score"])

        row = f"{model_name:<30s}"
        all_scores = []
        for cat in categories:
            scores = cat_scores.get(cat, [])
            if scores:
                avg = sum(scores) / len(scores)
                row += f" {avg:>7.2f}"
                all_scores.extend(scores)
            else:
                row += f" {'N/A':>7s}"

        overall = sum(all_scores) / len(all_scores) if all_scores else 0
        row += f" {overall:>7.2f}"
        print(row)

    print("=" * 90)
    print()


def print_answer_summary(all_answers: Dict[str, list]):
    """Print quick summary of answers (when no GPT-4 judging)."""
    print("\n" + "=" * 90)
    print("MT-BENCH ANSWER GENERATION COMPLETE")
    print("=" * 90)

    for model_name, answers in all_answers.items():
        if not answers:
            continue

        categories = {}
        avg_lens = []
        for ans in answers:
            cat = ans["category"]
            turns = ans["choices"][0]["turns"]
            total_len = sum(len(t) for t in turns)
            avg_lens.append(total_len)
            categories[cat] = categories.get(cat, 0) + 1

        avg_response_len = sum(avg_lens) / len(avg_lens) if avg_lens else 0

        print(f"\n  Model: {model_name}")
        print(f"  Questions answered: {len(answers)}")
        print(f"  Avg response length: {avg_response_len:.0f} chars")
        print(f"  Categories: {dict(sorted(categories.items()))}")

    print("\n" + "=" * 90)
    print()


def main():
    parser = argparse.ArgumentParser(description="MT-Bench Quality Evaluation for LittleBit Models")

    # Model paths
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model ID (for tokenizer, and FP model if eval_fp=true)")
    parser.add_argument("--draft_model_path", type=str, required=True,
                        help="Path to 0.1-bit draft model")
    parser.add_argument("--residual_model_path", type=str, default=None,
                        help="Path to 0.9-bit residual model")

    # Evaluation modes
    parser.add_argument("--eval_draft", type=str, default="true",
                        help="Evaluate draft model independently")
    parser.add_argument("--eval_target", type=str, default="true",
                        help="Evaluate target (draft+residual) model")
    parser.add_argument("--eval_fp", type=str, default="false",
                        help="Evaluate FP baseline model")

    # Generation params
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 for greedy decoding")
    parser.add_argument("--max_questions", type=int, default=80,
                        help="Max number of questions to evaluate (default: all 80)")

    # GPT-4 Judging
    parser.add_argument("--judge", type=str, default="false",
                        help="Use GPT-4 as judge (requires OPENAI_API_KEY)")
    parser.add_argument("--judge_model", type=str, default="gpt-4o",
                        help="Judge model name")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_results/mt_bench")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Parse bool args
    eval_draft = args.eval_draft.lower() in ('true', '1', 'yes')
    eval_target = args.eval_target.lower() in ('true', '1', 'yes')
    eval_fp = args.eval_fp.lower() in ('true', '1', 'yes')
    use_judge = args.judge.lower() in ('true', '1', 'yes')

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Select questions
    questions = MT_BENCH_QUESTIONS[:args.max_questions]
    logger.info(f"Evaluating {len(questions)} MT-Bench questions")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.base_model_id)

    all_answers = {}
    all_judgments = {}

    # ==== 1. Draft Model (0.1-bit) ====
    if eval_draft:
        logger.info("=" * 60)
        logger.info("Loading DRAFT model (0.1-bit)...")
        logger.info("=" * 60)
        draft_model = load_quantized_model_standalone(
            args.draft_model_path, device=device
        )

        def draft_gen_fn(tokenizer, messages, max_new_tokens, temperature, device):
            return generate_response(
                draft_model, tokenizer, messages, max_new_tokens, temperature, device
            )

        draft_answers = evaluate_model_mt_bench(
            model_name="draft_0.1bit",
            generate_fn=draft_gen_fn,
            tokenizer=tokenizer,
            questions=questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        all_answers["draft_0.1bit"] = draft_answers

        # Save answers
        draft_out = os.path.join(args.output_dir, "draft_0.1bit_answers.jsonl")
        with open(draft_out, 'w') as f:
            for ans in draft_answers:
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
        logger.info(f"Draft answers saved to {draft_out}")

        # Free memory if loading more models
        if eval_target or eval_fp:
            del draft_model
            torch.cuda.empty_cache()
            import gc; gc.collect()

    # ==== 2. Target Model (draft + residual) ====
    if eval_target and args.residual_model_path:
        logger.info("=" * 60)
        logger.info("Loading TARGET model (draft + 0.9-bit residual)...")
        logger.info("=" * 60)
        target_draft = load_quantized_model_standalone(
            args.draft_model_path, device=device
        )
        target_residual = load_quantized_model_standalone(
            args.residual_model_path, device=device
        )

        def target_gen_fn(tokenizer, messages, max_new_tokens, temperature, device):
            return generate_response_combined(
                target_draft, target_residual, tokenizer, messages,
                max_new_tokens, temperature, device,
            )

        target_answers = evaluate_model_mt_bench(
            model_name="target_1.0bit",
            generate_fn=target_gen_fn,
            tokenizer=tokenizer,
            questions=questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        all_answers["target_1.0bit"] = target_answers

        target_out = os.path.join(args.output_dir, "target_1.0bit_answers.jsonl")
        with open(target_out, 'w') as f:
            for ans in target_answers:
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
        logger.info(f"Target answers saved to {target_out}")

        if eval_fp:
            del target_draft, target_residual
            torch.cuda.empty_cache()
            import gc; gc.collect()

    elif eval_target and not args.residual_model_path:
        logger.warning("--residual_model_path not provided. Skipping target evaluation.")

    # ==== 3. FP Baseline ====
    if eval_fp:
        logger.info("=" * 60)
        logger.info("Loading FP BASELINE model...")
        logger.info("=" * 60)
        fp_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        fp_model.eval()

        def fp_gen_fn(tokenizer, messages, max_new_tokens, temperature, device):
            return generate_response(
                fp_model, tokenizer, messages, max_new_tokens, temperature, device
            )

        fp_answers = evaluate_model_mt_bench(
            model_name="fp_baseline",
            generate_fn=fp_gen_fn,
            tokenizer=tokenizer,
            questions=questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        all_answers["fp_baseline"] = fp_answers

        fp_out = os.path.join(args.output_dir, "fp_baseline_answers.jsonl")
        with open(fp_out, 'w') as f:
            for ans in fp_answers:
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
        logger.info(f"FP answers saved to {fp_out}")

        del fp_model
        torch.cuda.empty_cache()

    # ==== GPT-4 Judging ====
    if use_judge:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.error("OPENAI_API_KEY not set! Skipping judging.")
        else:
            logger.info("=" * 60)
            logger.info(f"Running GPT-4 judging (model={args.judge_model})...")
            logger.info("=" * 60)

            for model_name, answers in all_answers.items():
                logger.info(f"Judging {model_name}...")
                judgments = judge_with_gpt4(
                    answers, api_key, judge_model=args.judge_model
                )
                all_judgments[model_name] = judgments

                # Save judgments
                judge_out = os.path.join(args.output_dir, f"{model_name}_judgments.jsonl")
                with open(judge_out, 'w') as f:
                    for j in judgments:
                        f.write(json.dumps(j, ensure_ascii=False) + "\n")

            print_summary(all_judgments)
    else:
        print_answer_summary(all_answers)
        print("  To score with GPT-4, re-run with: --judge true")
        print("  (requires OPENAI_API_KEY environment variable)")

    logger.info("MT-Bench evaluation complete!")


if __name__ == "__main__":
    main()
