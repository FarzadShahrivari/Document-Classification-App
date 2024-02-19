# Document-Classification-App
This project aims to build a streamlined identity document classification system. Data is preprocessed and modeled using PyTorch (Lightening). A React-based UI allows image uploads, processed by a backend service for classification. Finally, Docker is used for deployment efficiency.
## How to Run the Application
**Step 1**: Begin by downloading all the contents from the GitHub repository.

**Step 2**: Next, navigate to the following link to download the "myModel.ckpt" file. Once downloaded, move the file to the "Document-Classification-App/Backend" directory. Ensure that the "myModel.ckpt" file is placed alongside "Backend.py", Dockerfile, and requirements.txt within the same directory. [Download Link](https://1drv.ms/u/s!AtuVVOX-wCJagpY_no1sQoIlLqvwYw)

**Step 3**: Make sure to have Docker Desktop running on your system (applicable for both Windows and Mac). Open a terminal and navigate to the location where you've stored the "Document-Classification-App" folder using the following command:

```setup
cd "path to the folder"/Document-Classification-App
```
Then, execute the command below in the terminal:

```setup
docker-compose up
```
**Step 4**: Open a web browser and go to ``http://localhost:5173''.

**Step 5**: Finally, sit back and enjoy using the machine learning application!
## Repository Structure

| Path | Description
| :--- | :----------
| master | The main folder containing the repository.
| &ensp;&ensp;&boxvr;&nbsp; [Backend](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Backend) | The backend service is responsible for receiving uploaded images from the UI and leveraging the trained ML model to predict the class, after which it sends the results back to the UI. This component is written in Python using Flask.
| &ensp;&ensp;&boxvr;&nbsp; [Data-Preprocessing-and-Model-Building](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Data-Preprocessing-and-Model-Building) | This component utilizes a deep neural network structure to classify documents, including ID cards and passports from various countries. It is implemented in Python using PyTorch with PyTorch Lightning framework.
| &ensp;&ensp;&boxvr;&nbsp; [Documents](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Documents) | This component encompasses comprehensive documentation and explanations detailing the approach employed in the document classification application process.
| &ensp;&ensp;&boxvr;&nbsp; [Frontend](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Frontend) | This component enables users to upload a test image from the dataset. Upon submission, the image is sent to the backend service for classification. This part is implemented in JavaScript using React.
| &ensp;&ensp;&boxur;&nbsp; docker-compose.yml | This Docker Compose configuration orchestrates, manages, and runs all these services together, ensuring the seamless operation of the document classification app.
## Final Note
Instead of following step 2, you can execute the Python file (PassportClassification.py) located in the "Data Preprocessing and Model Building" folder of the repository. Running this file is straightforward as all necessary Docker files and required libraries are included in the folder. Yet, detailed instructions for running this file can be found in the PDF document located in the "Documents" folder of the repository under the "Data Preprocessing and Model Building" section of the PDF, within the final notes. Please note that running this application can be moderately resource-intensive, as indicated in the recommended hardware requirements and processing time mentioned in the final paragraph of the introduction section of the PDF. After executing the Python file, navigate to the location where the Python file is located (PassportClassification.py along with the Dockerfile and requirements.txt). You will find a newly created folder named "PassportClassification." Inside this folder, there will be a folder named "checkpoints." Within this folder, locate a file with the .ckpt extension, typically starting with "epoch==." Rename this file to "myModel.ckpt" and proceed with the next steps to run the application.
