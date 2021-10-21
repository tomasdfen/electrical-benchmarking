function downloadPerformance(model){
    console.log(`Modelo: ${model}`)
    $.ajax({
        url: `/downloadPerformance/${model}`,
        type: "get",
        success: function (response){
            console.log(response)
            window.location = response
        },

        error:  function (xhr) {
            console.log("Hubo un error :/")
            console.log(xhr)
        }
    })
}

function downloadPreds(model){
    console.log(`Modelo: ${model}`)
    $.ajax({
        url: `/downloadPreds/${model}`,
        type: "get",
        success: function (response){
            console.log(response)
            window.location = response +`?nocache=${(new Date()).getTime()}`;
        },

        error:  function (xhr) {
            console.log("Hubo un error :/")
            console.log(xhr)
        }
    })
}

function gamma_num(){
    $("#gamma").change(function () {
        var numeric = (($(this).val()) === 'num') ? false : true;
        $("#gamma_num").prop("disabled", numeric);
    })

}

var nvars = 0
function validate () {
    checked = $("input[type=checkbox]:checked").length;

    if(!checked) {
        alert("Selecciona al menos un modelo");
        return false;
    }
    return true

};

function addrow(variables) {
    var featuresDiv = $("#features")[0] 
    var div = $.parseHTML(`<div class="row" id="${nvars}-var"></div>`)[0]
    var options = $.parseHTML('<div class="col row"></div>')[0]
    var options_up = $.parseHTML('<div class="row"></div>')[0]
    var options_down = $.parseHTML('<div class="row"></div>')[0]
    var options_down_left = $.parseHTML('<div class="col"></div>')[0]
    var options_down_right = $.parseHTML('<div class="col"></div>')[0]
    var select = $.parseHTML('<div class="col"><select class="form-select col" name="selectedvars"></select></div>')[0]
    console.log(variables)
    variables.forEach(column => {
        select.firstChild.append($.parseHTML(`<option value="${column}">${column}</option>`)[0])
    });

    options_up.append($.parseHTML(`<input type="number" id="${nvars}-back" name="${nvars}-back" class="form-control" pattern="((\d+)(\s)?[,|-]?+){1,}" required/>`)[0])
    
    options_down_left.append($.parseHTML(`<label for="" class="form-check-label">Tipo de lags</label>`)[0])
    options_down_left.append($.parseHTML(`<div class="form-check"><label for="${nvars}-continous" class="form-check-label">Continuo</label> \
                                            <input type="radio" class="form-check-input" name="${nvars}-backtype" id="${nvars}-continous" value="continous" onclick="$('#${nvars}-back')[0].type='number'" checked>
                                         </div>`)[0])
    options_down_left.append($.parseHTML(`<div class="form-check"><label for="${nvars}-selective" class="form-check-label">Selectivo</label>\
                                            <input type="radio" class="form-check-input" name="${nvars}-backtype" id="${nvars}-selective" value="selective" onclick="$('#${nvars}-back')[0].type='text'"> \
                                         </div>`)[0])
    
    options_down_right.append($.parseHTML(`<label for="" class="form-check-label">Transformación</label>`)[0])
    options_down_right.append($.parseHTML(`<div class="form-check"><label for="${nvars}-none" class="form-check-label selected">Ninguna</label> \
                                           <input type="radio" class="form-check-input" name="${nvars}-transformation" id="${nvars}-none" value="none">
                                           </div>`)[0])
    options_down_right.append($.parseHTML(`<div class="form-check"><label for="${nvars}-mean" class="form-check-label">Media</label> \
                                           <input type="radio" class="form-check-input" name="${nvars}-transformation" id="${nvars}-mean" value="mean">
                                           </div>`)[0])
    options_down_right.append($.parseHTML(`<div class="form-check"><label for="${nvars}-median" class="form-check-label">Mediana</label> \
                                           <input type="radio" class="form-check-input" name="${nvars}-transformation" id="${nvars}-median" value="median">
                                           </div>`)[0])


    options_down.append(options_down_left, options_down_right)
    options.append(options_up, options_down)
    div.append(select, options)
    featuresDiv.append(div)
    nvars++
}
function removerow(){
    nvars--
    $(`#${nvars}-var`).remove()
}