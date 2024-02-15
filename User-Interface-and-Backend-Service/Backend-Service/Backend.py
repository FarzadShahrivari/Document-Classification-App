# Necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# myCNN instance is required for Pytorch to load the pre-trained model's structure
class myCNN(nn.Module):
   def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(0.05),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
            )
        self.lin = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(7*7*256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 10)
        )   
   def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        return x

# Loading the pre-trained model for inference 
Model = myCNN()
checkpoint = torch.load('C:/Users/oasis/source/repos/Backend/Backend/myModel.ckpt')
Model.load_state_dict(checkpoint['state_dict'])
Model.eval()

# Pre-processing is required to make the recieved images from the fronted app compatible to the Pytorch pre-trained model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),
])

#Making a dictionary for class predictions
mydict = {0:'ID Card of Albania', 1:'Passport of Azerbaijan', 2:'ID Card of Spain', 3:'ID Card of Estonia',
          4:'ID Card of Finland', 5:'Passport of Greece', 6:'Passport of Latvia', 7:'Internal passport of Russia',
          8:'Passport of Serbia', 9:'ID Card of Slovakia'}

# Seting up a Flask application instance and enabling Cross-Origin Resource Sharing (CORS)
# to allow communication with the frontend application running on port 5173 
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Define a route handler for the root endpoint ("/") that returns a simple message indicating successful connection to the backend
@app.route("/")
def home():
    return {"message": "Hello from backend"}

# Define a route handler for the "/upload" endpoint that handles HTTP POST requests.
# This function receives an uploaded file from the frontend, saves it to the server's file system
# in a directory named "uploads", and retrieves the file path for further processing.
@app.route("/upload", methods=['POST'])
def upload():
    file = request.files['file']
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file.save('uploads/' + file.filename)
    img_path = f"./uploads/{file.filename}"
    
    # Making predictions based on the recieved image uploaded on frontend 
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        output = Model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        
    # Check if the uploaded file already exists in the "uploads" directory.
    # If the file exists, remove it from the server's file system to avoid duplicates.
    if os.path.exists(f"./uploads/{file.filename}"):
        os.remove(f"uploads/{file.filename}")
    
    # Sending the predictions label to the frontend
    return jsonify({"message": mydict[predicted_class.item()]})

# Start the Flask application server on localhost (by default) and port 5000.
if __name__ == '__main__':
    app.run(host='localhost', debug=True)