# Generated by Django 3.2.9 on 2021-12-14 13:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_auto_20211214_1900'),
    ]

    operations = [
        migrations.AlterField(
            model_name='img',
            name='predited_img',
            field=models.ImageField(upload_to=''),
        ),
    ]
