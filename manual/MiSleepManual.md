# MiSleep

An efficient software for manually sleep staging

如果在使用过程中出现任何问题，请及时联系！

### Load data
![img_2.png](imgs/img_2.png)

1. Sampling rate
   + 由记录数据时机器的设置所决定。
2. Epoch length
   + 后续自动睡眠标注功能所需要的参数，目前只作窗口划分作用。
3. Number of channel
   + 传入的数据中通道的数目，会根据传入的数据（Select data file）进行判断匹配。
4. Time acquisition
   + 数据获取的时间，首次传入数据或创建label file 需要修改成对应的时间。将会被用作时间戳以及保存在label file内。
5. Select data file
   + 选择一个数据文件进行标注操作，目前只支持`.mat` 文件格式。
6. Set/Select label file
   + 选择label file，首次进行数据标注需要新建一个`.txt`文件后，再进行文件的选择。
7. Channel Data
   + 点击进行数据的校验，如果校验成功则打开另一个窗口进行数据的展示，如果失败会提示错误原因。


### MiSleep

![img_4.png](imgs/img_4.png)

主要分为三个部分，左边上半部分是数据展示区域，左边下半部分是睡眠阶段展示区域，右边是操作区域。

#### 数据展示区域
1. 第一个图是时频图，展示当前时间段内每秒各个信号的强度随时间的变化，颜色越红表示强度越大，越蓝表示强度越弱。可以根据右侧操作区域内的Colorbar percentile进行比例的调节。
2. 中间是各个通道的信号图，左侧有该通道的编号，以及该通道的y轴范围，底部是时间坐标轴（以秒为单位）。
3. 滑动条可以控制时间，左右两个`<` `>`可以每次滑动一个epoch length的长度；点击滑动条空白处可以跳转到下一个（或上一个）窗口。
4. 随意点击信号图中的位置，可以在当前位置（秒）的开始位置标记一个marker，其中marker的名称可以自己修改。

#### 睡眠阶段展示区域
各个时间点所属于的睡眠阶段，每一根竖红线代表一个marker的标签。

#### 操作区域
1. 设置某一个通道为数据展示区域时频图的默认通道，需要在channel的列表里选中一个然后点击`Default channel for time frequency`按钮。
2. 保存选中channel的数据（`Save selected channels`按钮）或者直接保存所有channel的数据（`Save`按钮）。
3. 对每个睡眠阶段的数据进行合并保存，可以仅保存选中的channel(`Merge as sleep stages`按钮)或直接保存所有的channel(`Merge`按钮)。
4. 选中一个或某几个channel进行展示或隐藏或删除。
5. 使用滤波器对选中的一个channel进行`高通`，`低通`或`带通`滤波，设定滤波的频率并点击`Filter`按钮或按下`F`键进行滤波，滤波会产生一个新的channel。
6. 调节选中的一个或多个channel的y轴显示范围。
7. 选中一个`start-end`区域，并选中一个channel，点击`Spectrum`按钮或按下`S`键，可以绘制选中数据的时频图以及频谱图。可分别保存图像。
8. 选中`Marker`进行单点标记，选中`Start-End`进行`start-end`区域的选取。
9. 选取`start-end`区域后
   + 对该区域进行分析，见第`7`点
   + 将该区域标记上某种标签，点击`Label`按钮或按`L`键，选择一个标签进行标记，标签名称可自定义。
   + 将该区域标记为某个睡眠阶段，点击`1:NREM`，`2:REM`，`3:Wake`或`4:INIT`标记为对应的睡眠阶段，同时可以使用数字快捷键`1`，`2`，`3`和`4`进行以上快捷操作
10. 将当前窗口的时间段标记为某一个睡眠阶段，只能由快捷键进行，具体对应如下
    + `Shift + 1`标记为`NREM`区域
    + `Shift + 2`标记为`REM`区域
    + `Shift + 3`标记为`Wake`区域
    + `Shift + 4`标记为`Init`
11. 选中`Auto scroll`多选框，在标记完某个睡眠阶段后，可以自动跳转到该睡眠阶段的结束位置。
12. 点击`Save labels`或按下`CTRL+S`快捷键，对进行过的操作进行保存。
13. 使用时间戳或秒绝对值可以直接跳转到对应的位置。
14. 使用下拉框可以选择常用的窗口时间显示大小。
15. 选中`Custom epoch`可以自定义窗口显示epoch的数量。
16. 标签保存即`Save labels`功能每5分钟会自动执行一次，再次还是提醒各位常进行标签的保存，以防软件闪退。

