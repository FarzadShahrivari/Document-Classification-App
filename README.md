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
| &ensp;&ensp;&boxvr;&nbsp; [Backend](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Backend) | Config files containing the settings of the diffusion models.
| &ensp;&ensp;&boxvr;&nbsp; [Data-Preprocessing-and-Model-Building](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Data-Preprocessing-and-Model-Building) | Dataset information.
| &ensp;&ensp;&boxvr;&nbsp; [Documents](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Documents) | DDPM library.
| &ensp;&ensp;&boxvr;&nbsp; [Frontend](https://github.com/FarzadShahrivari/Document-Classification-App/tree/main/Frontend) | Fast Autoaugment modules.
| &ensp;&ensp;&boxur;&nbsp; docker-compose.yml | Utility modules.
