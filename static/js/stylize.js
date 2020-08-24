document.body.style.backgroundColor = "#23262a";

function validateForm() {
    var empt = document.forms["form1"]["fileToUpload"].value;
    if (empt == "") {
        alert("Please upload an image");
        return false;
    }
    else {
        return true;
    }
}

function validateImage() {
    var formData = new FormData();

    var file = document.getElementById("fileToUpload").files[0];

    formData.append("Filedata", file);
    var t = file.type.split('/').pop().toLowerCase();
    if (t != "jpeg" && t != "jpg" && t != "png" && t != "bmp") {
        alert('Please select a valid image file');
        document.getElementById("img").value = '';
        return false;
    }
    else return true;
}

function readURL(input, imgClass, labelClass) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      reader.onload = function(e) {
        $('#'+imgClass).attr('src', e.target.result);
        document.getElementById(imgClass).style.display = 'block';
        document.getElementById(labelClass).style.color = '#FAA916';
        $(".welcome-text").empty();
      }
      reader.readAsDataURL(input.files[0]);
    }
  }
  
  $("#contentFile").change(function() {
    readURL(this, 'contentImg', 'contentLabel');
  });

  $("#styleFile").change(function() {
    readURL(this, 'styleImg', 'styleLabel');
  });
