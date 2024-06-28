configs_dir = 'configs/assistant_configs_all'

TEMPLATE = f"""assistant_name: "0.5acc_corr:betaPARAMONEaPARAMTWOb_incorr:betaPARAMTHREEaPARAMFOURb"
assistant_correctness:
  distribution_name: "bernoulli"
  params:
    p: 0.5
confidence_when_correct:
  distribution_name: "beta"
  params:
    a: PARAMONE
    b: PARAMTWO
confidence_when_incorrect:
  distribution_name: "beta"
  params:
    a: PARAMTHREE
    b: PARAMFOUR
confidence_expression: "percentage"
"""

correct_confidence_params = [(15, 2), (12, 3), (6, 2), (9, 4), (4, 2), (5, 5)]
incorrect_confidence_params = [(2, 15), (3, 12), (2, 6), (4, 9), (2, 4), (5, 5), (15, 2), (12, 3), (6, 2), (9, 4), (4, 2),]

for paramone, paramtwo in correct_confidence_params:
  for paramthree, paramfour in incorrect_confidence_params:
    #(paramone, paramtwo), (paramthree, paramfour) = param_set
    if paramone - paramtwo < paramthree - paramfour:
      continue
    output = TEMPLATE.replace('PARAMONE', str(paramone)).replace('PARAMTWO', str(paramtwo)).replace('PARAMTHREE', str(paramthree)).replace('PARAMFOUR', str(paramfour))
    filename = f"corr:beta{paramone}a{paramtwo}b_incorr:beta{paramthree}a{paramfour}b.yaml"
    with open(f"{configs_dir}/{filename}", 'w') as f:
        f.write(output)
    print(f"Created {filename}")
