// 绑定按钮点击事件
function bindCapthaBtnClick(){
    // $()可以获取某个按钮
    $("#captcha-btn").on("click",function (event){
       // $(this)相当于$("#captcha-btn")，this相当于当前函数
       var $this = $(this);
        // 获取name为email的input标签的值
       var email = $("input[name='email']").val();
       if(!email){
           alert("请先输入邮箱!");
           return;
       }
       // 通过js向url发送网络请求：ajax
       $.ajax({
           url: "/user/captcha",
           method: "POST",
           data:{
               "email":email
           },
           // res为/captcha的视图函数return的内容
           success: function(res){
                var code = res['code'];
                if(code == 200){
                    // 在倒计时期间取消点击事件
                    $this.off("click");
                    // 开始倒计时
                    var countDown = 60;
                    var timer = setInterval(function (){
                        countDown -= 1;
                        if(countDown>0){
                            $this.text(countDown+"秒后重新发送");
                        }else {
                            $this.text("获取验证码");
                            // 重新执行以下函数，重新绑定点击事件
                            bindCapthaBtnClick();
                            // 停止(清除)倒计时，否则会一直执行下去
                            clearInterval(timer);
                        }
                    },1000);
                    alert("验证码发送成功！");

                }else{
                    alert(res['message']);
                }

           }

       })

    });

}

// $()中函数会等待网页文档所有元素加载完后才执行
$(function(){
    bindCapthaBtnClick();
})

