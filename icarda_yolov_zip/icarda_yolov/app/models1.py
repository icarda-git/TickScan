from django.db import models
from .validators import validate_file_size

class Img(models.Model):
    img = models.ImageField(upload_to='image/', null = True, blank = True, validators=[validate_file_size])

    predited_img = models.ImageField( null = True, blank = True)

class visitor_detials(models.Model):
    visitor_ip_addr = models.CharField(null = True, blank = True, max_length=50)
    server_name = models.CharField(null = True, blank = True, max_length=50)
    server_port = models.CharField(null = True, blank = True, max_length=50)
    remote_host = models.CharField(null = True, blank = True, max_length=50)


