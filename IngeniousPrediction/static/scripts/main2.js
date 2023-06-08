function enableStartBtn2() {
    var mySelect1 = document.getElementById('projects');
    var mySelect2 = document.getElementById('algs');
    mySelect1.onchange = (event) => {
        mySelect2.onchange = (event) => {
        
            $('#startBtn').prop('disabled',false);
        }
    }
       //<----to remove  disable attr of the button
  }
  enableStartBtn2()