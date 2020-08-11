$(document).ready(function() {
  $('select').material_select();
   $('.sidenav-trigger').sidenav();

 });


function select_ops(cols)
{
  var newSelect1 = document.getElementById("select_indep");
  for(var i=0;i<cols.length;i++)
  {
   var opt = document.createElement("option");
   opt.value= i;
   opt.innerHTML =cols[i];

   //append it to the select element
   newSelect1.add(opt,undefined);

  }

  alert("Columns of uploaded csv file are:\n"+cols)
  var elmnt = document.getElementById("process");
  elmnt.scrollIntoView();

}



var input = document.getElementById("fileUpload");
input.addEventListener( 'change', Upload );
function Upload(event)
{
 var regex = /^([a-zA-Z0-9\s_\\.\-:])+(.csv|.txt)$/;
 var fileUpload = event.srcElement;

 if (regex.test(fileUpload.value.toLowerCase()))
 {
     if (typeof (FileReader) != "undefined")
     {
         var reader = new FileReader();
         reader.readAsText(fileUpload.files[0]);
         reader.onload = function (e) {
             //var table = document.createElement("table");
             var rows = e.target.result.split("\n");
             var cols = rows[0].split(",");
             select_ops(cols);
            }

     }

     else
     {
         alert("This browser does not support HTML5.");
     }
   }

   else
   {
       alert("Please upload a valid CSV file.");
   }
}
