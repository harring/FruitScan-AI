#Contributors: Erik, Patricia, Jonathan

# Import necessary packages
from django.core.management.base import BaseCommand
from fruitscanapp.models import FruitClassification, MLWeights, ModelWeights

class Command(BaseCommand):
    help = 'Add sample data to the database'

    def handle(self, *args, **options):
        # Add sample data for FruitClassification
        fruit_instance = FruitClassification(label=2, pixels={"red": 255, "green": 255, "blue": 255})
        fruit_instance.save()

        # Add sample data for MLWeights
        weights_instance = MLWeights(
            version="v1",
            weight={"w1": 0.5, "w2": 0.8, "w3": 0.2},
            weight_0=0.5,
            weight_1=0.8,
            weight_2=0.2
        )
        weights_instance.save()

        # Fetch model weights and save the result
        model_weights = ModelWeights(version="1", path='ModelWeights/fruitscan_model_weights_v1.h5')
        model_weights.save()

        # Inform user about the successful operation
        self.stdout.write(self.style.SUCCESS('Sample data added successfully'))
