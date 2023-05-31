import torch
def debug_output(tensor:torch.Tensor, tensor_name = "tensor" ,output_path="debug/demo01.txt"):
    with open(output_path, 'w') as f:
        f.write("\n--------\n")
        f.write(f"{tensor_name}:\n"+str(tensor.tolist()))
        
        