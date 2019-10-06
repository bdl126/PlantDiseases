$(document).ready(function(){

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#img')
                    .attr('src', e.target.result);
            };

            reader.readAsDataURL(input.files[0]);
        }
    }
    $('#imageUpload').change(function(){
        readURL(this)
    })
    $('#button_Classify').click(function(event){
        console.log()
        var fd = new FormData()
        var files = $('#imageUpload')
        var files1 = $('#imageUpload')[0]
        var files2 = $('#imageUpload')[0].files[0]
        fd.append('file',files2)
        $.ajax ({
            type:'POST',
            url: '/predict',
            data : fd,
            contentType : false,
            cache: false,
            processData: false,
            async: true,
            success: function (data){
                $('#prediction').text(data);
                console.log('ajax works!')
            }

        })
        
    })
});