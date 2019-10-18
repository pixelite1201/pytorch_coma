# Reimplementation of COMA in pytorch

This is the reimplementation of [COMA](https://github.com/anuragranj/coma) code in Pytorch.
Please follow the licensing rights of the authors if you use the code.
## Requirements
This code is tested on Pytorch versoin 1.3. Requirments can be install by running

      pip install -r requirements.txt
    
Install mesh processing libraries from [MPI-IS/mesh](https://github.com/MPI-IS/mesh). Note that the python3 version of mesh package library is needed for this.

## Train
To start the training, follow these steps
1. Download the registered data from the [Project Page](https://coma.is.tue.mpg.de/) 
2. Generate default config file by running.

    ` python config_parser.py`
3. Add following parameters to the default.cfg
    1. data_dir - path of dataset downloaded in step 1 (e.g. /home/pixelite1201/mesh_raw/)
    2. checkpoint_dir - path where checkpoints will be stored
    3. visual_output_dir - if visualize is set to True, then the visual output from evaluation will be stored here.
    4. checkpoint_file - if provided a checkpoint_file path, the script will load the parameters from the file and continue the training from there.
    #### Rest of the parameters can be adjusted according to the requirement
    #### Note that data_dir and checkpoint_dir can also be provided as command line option and will overwrite config options as follows
         python main.py --data_dir 'path to data' --checkpoint_dir 'path to store checkpoint'
4. Run the training by providing the split and split_term as follows

     ` python main.py --split sliced --split_term sliced `
     
     ` python main.py --split expression --split_term bareteeth `
     
     ` python main.py --split identity --split_term FaceTalk_170731_00024_TA `

     #### Note that if no split term is provided ‘sliced’ will be used as default

## Evaluation
To evaluate on test data you just need to set eval flag to true in default.cfg and provide the path of checkpoint file.

## Data preparation
Although when you run the training, the data preprocessing takes place if the data is not already there. But if you want you can prepare the data before running the training as explained below.

  1. Download the data from the [Project Page](https://coma.is.tue.mpg.de/)

         python data.py --split sliced --data_dir PathToData
         python data.py --split expression --data_dir PathToData 
         python data.py --split identity --data_dir PathToData 

#### Note that the pre_transform is provided which normalize the data before storing it. 
