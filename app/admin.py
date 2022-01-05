from django.contrib import admin
from .models1 import Img,visitor_detials

admin.site.site_header = 'Admin Panel'

admin.site.register(Img)
admin.site.register(visitor_detials)
