import torch 
import clip 
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device) 

toy_images = ["/Users/kalebnewman/Desktop/whisper_your_interest/images/car_toy.jpg","/Users/kalebnewman/Desktop/whisper_your_interest/images/elephant_toy.jpg", "/Users/kalebnewman/Desktop/whisper_your_interest/images/ocotpus.jpg", 
               "/Users/kalebnewman/Desktop/whisper_your_interest/images/wrench_toy.jpg"]
stock_images = ["/Users/kalebnewman/Desktop/whisper_your_interest/images/car_stock.jpeg","/Users/kalebnewman/Desktop/whisper_your_interest/images/elephant_stock.jpeg", "/Users/kalebnewman/Desktop/whisper_your_interest/images/octopus_stock.jpeg",
                 "/Users/kalebnewman/Desktop/whisper_your_interest/images/wrench_stock.jpeg" ]
toy_names = ["Car", "elephant", "octopus", "screwdriver"]

num_toys = len(toy_names)

stock_tensor = torch.zeros((num_toys, 512), device=device)
toy_tensor = torch.zeros((num_toys, 512), device=device)
names_tensor = torch.zeros((num_toys, 512), device=device)

i = 0
for x, y, z in zip(toy_images, stock_images, toy_names):
    stock_tensor[i] = model_clip.encode_image(preprocess(Image.open(y)).unsqueeze(0).to(device))
    toy_tensor[i] = model_clip.encode_image(preprocess(Image.open(x)).unsqueeze(0).to(device))
    names_tensor[i] = model_clip.encode_text(clip.tokenize(z).to(device))

    i+=1


compare_tensor = torch.zeros((num_toys, num_toys), device=device)

q = 0
for x in names_tensor:
    compare_tensor[q] = torch.cosine_similarity(stock_tensor, x)
    q+=1

# w=0

# while w < 4:
#     print(torch.argmax(compare_tensor[w]).item())
#     w+=1