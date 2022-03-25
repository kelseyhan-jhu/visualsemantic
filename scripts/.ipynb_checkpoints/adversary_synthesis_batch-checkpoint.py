import os

# for c in range(25):
#     for s in range(25):
#         style_index = str(s)
#         content_index = str(c)
#         command = "python scripts/adversary_synthesis.py --style_image face --style_index " + style_index + " --content_image scene --content_index " + content_index + " --content_weight 10 --num_steps 2500"
#         os.system(command)

for c in range(5):
    for s in range(c*2, c*2+5):
        style_index = str(s)
        content_index = str(c)
        command = "python scripts/adversary_synthesis.py --style_image scene --style_index " + style_index + " --content_image face --content_index " + content_index + " --content_weight 10 --num_steps 2500"
        os.system(command)

for c in range(5):
    for s in range(c*2, c*2+5):
        style_index = str(s)
        content_index = str(c)
        command = "python scripts/adversary_synthesis.py --style_image scene --style_index " + style_index + " --content_image object --content_index " + content_index + " --content_weight 10 --num_steps 2500"
        os.system(command)

for c in range(5):
    for s in range(c*2, c*2+5):
        style_index = str(s)
        content_index = str(c)
        command = "python scripts/adversary_synthesis.py --style_image object --style_index " + style_index + " --content_image face --content_index " + content_index + " --content_weight 10 --num_steps 2500"
        os.system(command)
