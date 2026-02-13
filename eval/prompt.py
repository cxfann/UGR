

sft_prompt = "{instruction}\n\n### Response:{response}"







all_prompt = {}


seqrec_prompt = []

prompt = {}
prompt["instruction"] = "The user has interacted with items {inters} in chronological order. Can you predict the next possible item that the user may expect?"
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

all_prompt["seqrec"] = seqrec_prompt

