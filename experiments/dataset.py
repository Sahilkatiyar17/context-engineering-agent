# experiments/dataset.py

EXPERIMENT_DATASET = [
    {
        "id": "q1",
        "question": "What is Proximal Policy Optimization (PPO) in reinforcement learning?",
        "purpose": "Pure search-heavy baseline. No memory dependency. Tests raw retrieval token cost.",
    },
    {
        "id": "q2",
        "question": "How does it compare to TRPO in terms of computational efficiency?",
        "purpose": "Requires short-term memory to resolve 'it' -> PPO. New search needed too.",
    },
    {
        "id": "q3",
        "question": "By the way, I personally prefer concise, bullet-point answers rather than long paragraphs.",
        "purpose": "A stated preference, not a question. Tests long-term memory EXTRACTION -- should get stored as a fact worth remembering.",
    },
    {
        "id": "q4",
        "question": "Can you explain the clipping objective used in PPO in more detail?",
        "purpose": "Needs short-term memory (PPO context) + new search. Also a qualitative check: does the answer respect the q3 preference?",
    },
    {
        "id": "q5",
        "question": "What are some recent developments or variants of PPO being discussed in the RL research community?",
        "purpose": "Search-heavy, recency-dependent, likely returns more/noisier results -- stress test for filtering and ranking.",
    },
    {
        "id": "q6",
        "question": "Summarize everything we've discussed about PPO so far, and remind me how I like my answers formatted.",
        "purpose": "Synthesis question -- exercises BOTH short-term (conversation summary) and long-term (q3's preference) memory at once. The hardest, most informative question in the set.",
    },
]