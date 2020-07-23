from django.db import models

class RunDB(models.Model):
   TASK = models.IntegerField()
   CSV_FILE = models.FileField(upload_to = 'database')
   DEP_VAR = models.IntegerField()

   class Meta:
      db_table = "run_db"
