export async function render() {
  const input_img = document.createElement("input");
  const input_preview = document.createElement("img");

  input_img.name = "input_img";
  input_img.type = "file";
  input_img.addEventListener("change", function() {showImage(input_img);});

  input_preview.id = "input_preview";

  document.body.appendChild(input_img);
  document.body.appendChild(input_preview);
}

function showImage(input)
{
  var reader;
  const input_preview = document.getElementById("input_preview");

  if (input.files && input.files[0]) {
    reader = new FileReader();

    reader.onload = function(e) {
      input_preview.setAttribute('src', e.target.result);
    }

    reader.readAsDataURL(input.files[0]);
  }
}
