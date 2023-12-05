from django.core.management.base import BaseCommand
from fruitscanapp.models import UploadedImage  
import os
from django.conf import settings

# THis method can be used to remove all the instances of uploaded image from the database
# Run the command python manage.py delete_images to erase them all

class Command(BaseCommand):
    help = 'Deletes all UploadedImage instances and their files'

    def handle(self, *args, **kwargs):
        for image_obj in UploadedImage.objects.all():
            if image_obj.image:
                image_path = os.path.join(settings.MEDIA_ROOT, image_obj.image.name)
                if os.path.isfile(image_path):
                    os.remove(image_path)

            image_obj.delete()
        self.stdout.write(self.style.SUCCESS('Successfully deleted all images and objects'))
