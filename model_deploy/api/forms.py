from django import forms

class RunForm(forms.Form):
    TASK = forms.IntegerField()
    CSV_FILE = forms.FileField()
    DEP_VAR = forms.IntegerField()

'''class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()'''
