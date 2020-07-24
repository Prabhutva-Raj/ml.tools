from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np

from django.template import loader, RequestContext
#from django.http import HttpResponseRedirect

# Create your views here.

@api_view(['GET'])
def index(request):
    #return_data = { "error" : "0",  "message" : "Successful" }
    template = loader.get_template('api/index.html')
    context = {}
    #return Response(template.render(context))
    return render(request, 'api/index.html', context=context)

'''created a simple index page, it’s more like a welcome page for every website or API,
notice the decorator used in line 9, what that decorator does it that it checks the API request method to see if it’s a
GET request being made, if it’s not a GET request then you won’t be able to access that function.'''

@api_view(['GET'])
def howtooperate(request):
    template = loader.get_template('api/HowToOperate.html')
    context = {}
    return render(request, 'api/HowToOperate.html', context=context)


from .forms import RunForm
from .models import RunDB
from .machinelearning_models import runmodels
from .functions import handle_uploaded_file

def run(request):
    if request.method == "POST":
        myRunForm = RunForm(request.POST, request.FILES)

    run_db = RunDB()
    if myRunForm.is_valid():
        run_db.TASK = myRunForm.cleaned_data["task"]
        run_db.CSV_FILE = myRunForm.cleaned_data["csvfile"]
        run_db.DEP_VAR = myRunForm.cleaned_data["dep_var"]
        handle_uploaded_file(request.FILES['csvfile'])
        run_db.save()
    else:
        myRunForm = RunForm()

    #is_private = request.POST.get('is_private', False)
    '''task = request.GET['task']
    dep_var = request.GET['dep_var']
    data_csv = request.GET['csvfile']'''


    task = request.POST.get('task')
    dep_var = request.POST.get('dep_var')
    data_csv = request.POST.get('csvfile')

    #accuracies = runmodels(run_db.CSV_FILE, run_db.DEP_VAR, run_db.TASK)
    outputs = runmodels(data_csv, dep_var, task)
    #raise Exception("Accuracies: ",accuracies)

    return  render(request, 'api/Results.html', context={'data':outputs})
