/*$(function() {
    $('#nav li a').click(function() {
        $(this).closest('li') // select the parent <li> tag
        .addClass('active')// add the active class
        .siblings() // select the siblings of the active tag
        .removeClass('active'); // remove the active class from the other <li>
    });
});*/

// dropdown
$(document).ready(function(){
    $('.ui.dropdown').dropdown();
});

// Semantic UI use the code below to initialiez the forms
$(document).ready(function(){
  // initialize the form an fields
  $('.value-to-reset')
  .form({
    fields: {
      fileInput:{
        identifier: 'name',
        rules: [
          {
            type : 'empty'
          }
        ]
      }
    }
  });
});

// Used to reset forms
function myOwnReset(){
  $('.value-to-reset').form('reset');
}

// Change gpu and cpu stats in train page
function get_train_stats(filename){
    $.ajax({
        type: 'POST',
        url: filename,
        dataType: 'html',
        success:function(data)
        {
            var start = data.indexOf('tliczeubgteipb') - 10;
            var end = data.indexOf('iprzhbgrzobgbvreo', start + 1) - 5;
            $('#training-progression-information').html(data.slice(start, end));

            var start = data.indexOf('abcdgfdqpjfzeqssd') + 24;
            var end = data.indexOf('abcdgfduipjfzeqssd', start + 1) - 5;
            $('#gpu_cpu_gauge').html(data.slice(start, end));
        },
        error: function (xhr, status, error) {
            console.log(error);
        }
    });
}

// launch a function immediately when the page is loaded and then call it with an interval
function setIntervalImmediately(func, interval) {
    func();
    return setInterval(func, interval);
}

// Get cpu and gpu stats each 2 seconds in trainin page
if (window.location.pathname == "/training_page" || window.location.pathname == "/train_standard_model"){
    setIntervalImmediately(function(){get_train_stats('/get_train_stats');}, 1000);
    //Script to train the server once dataset is uploaded.
    $(document).ready(
        function(){
            $('#train_server').attr('disabled',true);
            $('#upload_file_train').change(
                function(){
                    if ($(this).val()){
                        $('#trainupload').submit();
                    }
                }
            );
        }
    );
}


// Change gpu and cpu stats in train page
function get_download_stats(filename){
    $.ajax({
        type: 'POST',
        url: filename,
        dataType: 'html',
        success:function(data)
        {
            var start = data.indexOf('jklezffzelkj') + 24;
            var end = data.indexOf('fzelkjjklezf', start + 1) - 5;
            $('#download-bar').html(data.slice(start, end));
        },
        error: function (xhr, status, error) {
            console.log(error);
        }
    });
}

// Get download stats each 2 seconds in dataset page
if (window.location.pathname == '/dataset' || window.location.pathname == '/create_standard_dataset'){
    $(document).ready(function(){
        $('#download-bar').html('')
        setIntervalImmediately(function(){get_download_stats('/dataset');}, 1000);
    });
}

//initialize semantic ui checkboxes
$(document).ready(function(){
    $('.ui.checkbox').checkbox();
    $('.ui.radio.checkbox').checkbox();
});

// Hide warning message if click in x icon in dataset page
$(document).ready(function(){
    $('.hide-icon').click(function(){
        $('.warning-message').hide();
    });
});

// Hide warning message if click in x icon in dataset page
$(document).ready(function(){
    $('.hide-icon').click(function(){
        $('.warning-message').hide();
    });
});

// Elegant alert if dataset already exists in dataset page
$(document).ready(function(){
    $('.ui.basic.modal')
      .modal({
        blurring: true
      }).modal('show');
});

// Drag and drop image in detection page
$(document).ready(function(){
    // Basic
    $('.dropify').dropify();
    // Translated
    $('.dropify-fr').dropify({
        messages: {
            default: 'Drag and drop a file here or click',
            replace: 'Drag and drop a file or click to replace',
            remove:  'Delete',
            error:   'Sorry, this file is too big'
        }
    });
    // Used events
    var drEvent = $('#input-file-events').dropify();
    drEvent.on('dropify.beforeClear', function(event, element){
        return confirm('Do you really want to delete \'' + element.file.name + '\' ?');
    });
    drEvent.on('dropify.afterClear', function(event, element){
        alert('File deleted');
    });
    drEvent.on('dropify.errors', function(event, element){
        console.log('Has Errors');
    });
    var drDestroy = $('#input-file-to-destroy').dropify();
    drDestroy = drDestroy.data('dropify')
    $('#toggleDropify').on('click', function(e){
        e.preventDefault();
        if (drDestroy.isDropified()) {
            drDestroy.destroy();
        } else {
            drDestroy.init();
        }
    })
});

// Disable/enable submit button in dataset page
$(document).ready(function(){
    var checkboxes = $(':checkbox'),
        submitButt = $('#validateStandardDataset');
    checkboxes.change(function() {
        submitButt.attr('disabled', !checkboxes.is(':checked'));
    });
});

// Disable/enable submit button in train page
$(document).ready(function(){
    var checkboxes = $(':radio.dataset-train-check'),
        submitButt = $('#validateStandardModel');
    checkboxes.change(function() {
        submitButt.attr('disabled', !checkboxes.is(':checked'));
    });
});

// Disable submit image detection until an url is provided
$(document).ready(function(){
    $('#classifyurl').attr('disabled',true);
    
    $('#imageurl').keyup(function(){
        if($(this).val().length != 0 && $('.dataset-train-check').is(':checked')){
            $('#classifyurl').attr('disabled', false);
        }
        else
        {
            $('#classifyurl').attr('disabled', true);        
        }
    });

    $('.dataset-train-check').change(function(){
        if($('#imageurl').val().length != 0 && $(this).is(':checked')){
            $('#classifyurl').attr('disabled', false);
        }
        else
        {
            $('#classifyurl').attr('disabled', true);        
        }
    });
});

// Quick example write url in input
$(document).ready(function(){
    $('#quick-example').change(function(){
        $('#imageurl').val($('#quick-example').find(":selected").val());
        if($('.dataset-train-check').is(':checked')){
            $('#classifyurl').attr('disabled', false);
        }
        else
        {
            $('#classifyurl').attr('disabled', true);        
        }
    });
});


// Disable submit image detection until a file si provided
$(document).ready(function(){
    $('#validateOwnImage').attr('disabled',true);
    
    $('#imagefile').change(function(){
        if($(this).val().length != 0 && $('.dataset-train-check').is(':checked')){
            $('#validateOwnImage').attr('disabled', false);
        }
        else
        {
            $('#validateOwnImage').attr('disabled', true);        
        }
    });

    $('.dataset-train-check').change(function(){
        if($('#imagefile').val().length != 0 && $(this).is(':checked')){
            $('#validateOwnImage').attr('disabled', false);
        }
        else
        {
            $('#validateOwnImage').attr('disabled', true);        
        }
    });
});

// Dimmer content in home page
$(document).ready(function() {
  $('.ui.message.blurring.dimmable').dimmer({on: 'hover'});
});

// Pop up showing informations about Mnist datasets when hover
$(document).ready(function() {
    $('#dataset-mnist-checkbox').popup({
        position : 'right center',
        html    : "<div class='header'>Mnist dataset</div>\
        <ul><li><strong>Download time</strong>: ~ 1 second</li><li><strong>Number of images</strong>: 60.000</li><li><strong>Size</strong>: 20x20</li></ul>"
    });
});

// Pop up showing informations about Cifar10 datasets when hover
$(document).ready(function() {
    $('#dataset-cifar10-checkbox').popup({
        position : 'right center',
        html    : "<div class='header'>Cifar 10 dataset</div>\
        <ul><li><strong>Download time</strong>: ~ 1 minute</li><li><strong>Number of images</strong>: 60.000</li><li><strong>Size</strong>: 32x32</li></ul>"
    });
});

// Pop up showing informations about Mnist training when hover
$(document).ready(function() {
    $('#train-mnist-checkbox').popup({
        position : 'right center',
        html    : "<div class='header'>Mnist training</div>\
        <ul><li><strong>Train with CPU</strong>: ~ 45 minutes</li><li><strong>Train with GPU</strong>: ~ 30 seconds</li></ul>"
    });
});

// Pop up showing informations about Cifar10 quick training when hover
$(document).ready(function() {
    $('#train-cifar10-quick-checkbox').popup({
        position : 'right center',
        html    : "<div class='header'>Cifar 10 quick training</div>\
        <p>This is a quick training based on CIFAR10 dataset. The server will be quicker to get trained but will be less accurate.\
        <ul><li><strong>Train with CPU</strong>: ~ 1h30m</li><li><strong>Train with GPU</strong>: ~ 3 minutes</li></ul>"
    });
});

// Pop up showing informations about Cifar10 full training when hover
$(document).ready(function() {
    $('#train-cifar10-full-checkbox').popup({
        position : 'right center',
        html    : "<div class='header'>Cifar 10 full training</div>\
        <p>This is a full training based on CIFAR10 dataset. The server will be slower to get trained but will be more accurate.\
        <ul><li><strong>Train with CPU</strong>: ~ 21 hours</li><li><strong>Train with GPU</strong>: ~ 40 minutes</li></ul>"
    });
});

//Show and hide forms depending on the checkbox which is checked in detection page
$(document).ready(function(){
    $("#form_local").hide();
    $("#local-radio").change(function() {
        if($(this).is(":checked")) {
            $("#form_url").hide();
            $("#form_local").show();
        }
    });
    $("#url-radio").change(function() {
        if($(this).is(":checked")) {
            $("#form_local").hide();
            $("#form_url").show();
        }
    });
});