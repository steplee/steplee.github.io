$(window).scroll(function(e){
    parallax();
});

function parallax(){
    var scrolled = $(window).scrollTop();
      $('#index-outer').css('background-position-y',-(465+scrolled*1.25)+'px'); // here you scroll downward
}