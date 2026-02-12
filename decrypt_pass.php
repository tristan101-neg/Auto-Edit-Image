<?php

$hash = '$2y$10$DpVFuD2FdFm3lJE/nK6q3upUBY1uyTPKzyztVSQ2IQO7G5tJUUMMW';
$password = 'gary123';

if (password_verify($password, $hash)) {
    echo "Password is correct!";
} else {
    echo "Password is incorrect!";
}
?>