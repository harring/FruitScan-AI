# Contributors: Erik, Mijin, Patricia, Jonathan

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
| Research Django/SQLite                 | All                 | Yes             | -             |   -                 |

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

| Week 4 (Week 48)                       |                     |              |              |                    |
|----------------------------------------|---------------------|--------------|--------------|--------------------|
|                                        | Who was responsible |  Delivered   |  Integrated  |  Notes             |
| Create user model | Mijin               | Yes          | Yes          | -                  |
| User registration and login functionality | Mijin               | Yes          | Yes          | -                  |
| Front-end development in user-related pages | Mijin               | Yes          | Yes          | -                  |
| Save and display user prediction history | Mijin               | Yes          | Yes          | -                  |
| Recreate database | Mijin               | Yes          | Yes          | -                  |
| Move Dataset to Database | Erik, Jonathan | Yes | Yes | Stores the images used for training in the database as binary |
| Trained models get confusion matrix | Erik, Jonathan | Yes | Yes | Each new model trained will automatically get a confusion matrix generated. |
| Add/remove/update datasets from admin panel | Erik, Jonathan | Yes | Yes | Admins can change/update the datasets used for training and testing |
| Admin can deploy chosen version | Erik, Jonathan | Yes | Yes | Admins can change which version of the model is deployed(used by users when predicting)  |
| Admin get model evaluation | Erik, Jonathan | Yes | Yes | Admins can get performance data on chosen model |
| Admin panel design and error handling | Erik, Jonathan | Yes | Yes | Various improvements to the admin panel |
| Remove title from uploaded image | Patricia | Yes | Yes | Change the model and the form so we sabe only the image in the database. |
| Django user panel | Patricia | Yes | Yes | Modified the front-end to match the mock-up sketch | - |
| Delete uploaded image after prediction if not logged in | Patricia | Yes | Yes | Delete the image from the database and the media folder. Created a code that clean all the images from the database as well.|
| Resize uploaded image to same size to display in the prediction page | Patricia | Yes | Yes | - |
| Add fruit nutritional information | Patricia | Yes | Yes | - |
| Improvements on user GUI | Patricia | Yes | Yes | Add style to all front-end pages, background, logo.| - | 

| Week 5 (Week 49)                       |                     |              |              |                    |
|----------------------------------------|---------------------|--------------|--------------|--------------------|
|                                        | Who was responsible |  Delivered   |  Integrated  |  Notes             |
| Migrate to GitHub | All | Yes|Yes| Manually migrate whole project to GitHub due to Chalmers IT problems with GitLab|
| Finalize dockerization | Mijin | Yes | Yes | - |
| Update section 1, 2 in the report | Mijin | Yes | Yes | - |
| Create unit test | Mijin, Erik | Yes| Yes | Integrated with git action pipeline |
| Add git action pipeline | Erik | Yes | Yes | To allow for k8 deployment and unit testing |
| Deployment with kubernetes and Google Cloud | Erik, Jonathan | Yes | No |  Deployed K8 branch to google cloud kubernetes cluster|
|Reroute different admin related things  | Jonathan |Yes|Yes|-|
|Protect the admin panel from not authorised users|Jonathan|Yes|Yes|-|
|Create explainability GUI and style|Patricia|Yes|Yes|Also styled the buttons in all user pages|
|Create explainability plots using LIME|Patricia|Yes|Yes|-|
|Show percentage of predictions in the explainability page and text for limitations|Patricia|Yes|Yes|-|
|Add spinning thing for loading in the explainability |Patricia|Yes|Yes|-|

| Week 6 (Week 50)                       |                     |              |              |                    |
|----------------------------------------|---------------------|--------------|--------------|--------------------|
|                                        | Who was responsible |  Delivered   |  Integrated  |  Notes             |
|Prepare for presentation|All|Yes|Yes|-|
|Make presentation at the fair|All|Yes|Yes|-|
|Various improvements to the application|All|Yes|Yes|Testing hyper parameters, admin panel usability, loading indicators|
|Write assignment 2 report|All|No|No|Work in progress|

| Week 7 (Week 51)                       |                     |              |              |                    |
|----------------------------------------|---------------------|--------------|--------------|--------------------|
|                                        | Who was responsible |  Delivered   |  Integrated  |  Notes             |
|Write assignment 2 report|All|Yes|Yes|-|
|Record video about cloud deployment|All|Yes|Yes|-|
|Various improvements to application and repository|All|Yes|Yes|Explainability shows percentage, confusion matrix shows text labels, readme file|
|Make final submission|All|Yes|Yes|-|
