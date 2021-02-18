CREATE USER 'dstaapp'@'%' IDENTIFIED BY 'dstaapp';
GRANT ALL PRIVILEGES ON *.* TO 'dstaapp'@'%';
CREATE DATABASE intel_sample;
USE intel_sample;
source datay.sql
