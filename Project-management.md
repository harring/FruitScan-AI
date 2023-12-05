# Responsibilities
| Week 1 (Week 45)                       |                     |              |              |                    |
|----------------------------------------|---------------------|--------------|--------------|--------------------|
|                                        | Who was responsible |  Delivered   |  Integrated  |  Notes             |
| Set up kanban board on Trello          | All                 | Yes          | -            | -                  |
| Set up a project repository            | All                 | Yes          | -            | -                  |
| Writing draft for assignment 1         | All                 | Yes          | -            | -                  |
| Setup development environment (conda)  | All                 | Yes          | -            | -                  |

| Week 2 (Week 46)                       |                     |              |              |                    |
|----------------------------------------|---------------------|--------------|--------------|--------------------|
|                                        | Who was responsible |  Delivered   |  Integrated  |  Notes             |
| Create team contract                   | All                 | Yes          | -            | -                  |
| Create toy model                       | All                 | Yes          | -            | -                  |
| Finalize Assignment 1 for submission   | All                 | Yes          | -            | -                  |
| Research Django/SQLite                 | All                 |              |              |                    |

| Week 3 (Week 47)                       |                     |              |              |                    |
|----------------------------------------|---------------------|--------------|--------------|--------------------|
|                                        | Who was responsible |  Delivered   |  Integrated  |  Notes             |
| Create Django model(schema) | Mijin               | Yes          | Yes          | -                  |
| Create Django app into docker image | Mijin               | Yes           | No           | Integration of docker-related feature will be made at later phase.  |
| Separate database from docker container | Mijin | No |No| Work in progress for docker-compose file to keep the stable database not impacted by a docker container.
| Create UI Mockup   | Jonathan  | Yes          | Yes            | -                  |
| Create Admin panel  | Jonathan |  Yes           |   Yes           |  Not connected to any model functionality as of now |
| Reroute built in admin login to custom admin panel | Jonathan |Yes|Yes|-|
|Look closer at datasets and pick one|Erik|Yes|Yes|Looked closer at the datasets we found and decided which one to use.|
|Add part of dataset resized to the repository|Erik|Yes|Yes|Since we are required to deploy the training to the cloud we are only able to use part of the dataset, so it was reduced to 600 images and resized to 256x256|
|Train new model on website using dataset with versioning|Erik|Yes|Yes|It is possible to train models on the django website, they are created with a version number following the last version created. If more images are added these are automatically included in the training of the new model.|
|Create confusion matrix and plot cnn when training on website|Erik|Yes|Yes|When training a new model a confusion matrix and a plot of the CNN is also generated and saved, these will have matching version number to the model they belong to|
|Setup database (SQLite)|Patricia|Yes|Yes|Made a page to test retrieving database information that were saved. Should be modified later on.|
|Upload picture feature|Particia|Yes|Yes|Layout will be changed later on.|
|Turn uploaded image into data|Particia|Yes|Yes|Convert image loaded into same resolution used in the model.|
|Integrate model with django|Particia|Yes|Yes|Load model to django and make prediction. Display the result in the diaplay_page, which will need to be changed later on|
|Save and load only weights instead of full model (keras file)|Particia|Yes|Yes|Change way of saving model to save only weights. Load only weights file (.h5) into django. Create a model architecture in django to be used with the weights for prediction.|
