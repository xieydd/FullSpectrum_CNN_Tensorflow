import os
files = os.listdir("H:/bearing data/bearing_IMS/1st_test_new")#列出当前目录下所有的文件

for filename in files:
    portion = os.path.splitext(filename)#分离文件名字和后缀
    #print(filename)
    #if portion[1] !=".01":#根据后缀来修改,如无后缀则空
    #newname = filename+".01"#要改的新后缀
    os.chdir("H:/bearing data/bearing_IMS/1st_test_new")#切换文件路径,如无路径则要新建或者路径同上,做好备份
    os.rename(filename,portion[0])