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
    $.ajax({
      async: true,
      url: '/reload_indexes',
      method: 'GET',
      processData: false,
      contentType: false,
      success: function (result) {
        $('#index').attr("disabled", false)
      },
      error: function (error) {
        $('#index').attr("disabled", false)
      }
    })
}

function open_folder() {
    $('#index').attr("disabled", true)
    $.ajax({
      async: true,
      url: '/open_folder',
      method: 'GET',
      processData: false,
      contentType: false,
      success: function (result) {
        $('#index').attr("disabled", false)
      },
      error: function (error) {
        $('#index').attr("disabled", false)
      }
    })
}