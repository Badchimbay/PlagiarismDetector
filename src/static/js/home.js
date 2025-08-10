function send(form) {
    $('#result')[0].innerHTML = 'Загрузка...'
    $.ajax({
      async: true,
      url: '/',
      method: 'POST',
      data: new FormData(form),
      dataType: 'json',
      processData: false,
      contentType: false,
      success: function (result) {
        $('#result')[0].innerHTML = result.message
        if (result.image) {
            $('#img img')[0].src = result.image
        }
      },
      error: function (error) {
        $('#result')[0].innerHTML = 'Ошибка!'
        console.log(error)
      }
    })
}

function reload_indexes() {
    $('#index').attr("disabled", true)
    $('#index i').addClass('loading')
    $.ajax({
      async: true,
      url: '/reload_indexes',
      method: 'GET',
      processData: false,
      contentType: false,
      success: function (result) {
        //
      },
      error: function (error) {
        //
      },
      complete: function () {
        $('#index').attr("disabled", false)
        $('#index i').removeClass('loading')
      }
    })
}

function sendModalFiles(form) {
    var button = $('#submitIndexFile');
    button[0].innerText = 'Загрузка...'
    button.attr("disabled", true)
    $.ajax({
      async: true,
      url: '/upload',
      method: 'POST',
      data: new FormData(form),
      dataType: 'json',
      processData: false,
      contentType: false,
      success: function (result) {
        //
      },
      error: function (error) {
        console.log(error)
      },
      complete: function () {
        var button = $('#submitIndexFile');
        button.attr("disabled", false)
        button[0].innerText = 'Загрузить'

        var modal = bootstrap.Modal.getInstance('#uploadModal');
        modal.hide();
      }
    })
}