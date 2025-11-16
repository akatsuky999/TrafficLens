## TrafficLens

### Overview

TrafficLens is a Python-based desktop application for interactive exploration of traffic trajectory data.  
It provides an Excel-like GUI built with Tkinter, plus a clean separation between GUI and data/analysis logic.

TrafficLens is designed for:

- Importing large CSV traffic datasets with fixed columns
- Searching, sorting and filtering trip records
- Basic descriptive statistics and publication-quality plots
- Spatio-temporal aggregation (time × node flow matrices)
- Exporting processed results for downstream analysis

#### Data schema

Each row in the input CSVs represents one trip (or detection record) with the following columns:

1. `VehicleType` (categorical)
2. `DetectionTime_O` (departure time)
3. `GantryID_O` (origin gantry / node code)
4. `DetectionTime_D` (arrival time)
5. `GantryID_D` (destination gantry / node code)
6. `TripLength` (numeric, e.g. distance or duration)
7. `TripEnd` (trip end code)
8. `TripInformation` (free text)

These columns are treated as different logical types (time, code, numeric, text) to drive plotting and filtering.

#### Main features

- **File**
  - Import a single CSV file
  - Import all CSVs from a folder
  - Merge additional files/folders into the current dataset
  - Export the current result (full table or spatio-temporal matrix) to CSV, XLSX or NPY
  - Clear all data and restart analysis

- **Operations**
  - Keyword search with fuzzy or strict matching
  - Global search across all string columns or column-specific search
  - Display match count and highlight matched rows in the table
  - Column-based sorting (ascending / descending) using the column selected in the table
  - Pagination controls for large datasets (jump to page, next/previous)

- **View**
  - Field-based filtering on:
    - `DetectionTime_O`, `DetectionTime_D` (time range)
    - `VehicleType` (category subset)
    - `TripLength` (numeric range)
  - Dynamic UI controls depending on the selected field type
  - Descriptive statistics and in-window plotting for a selected field:
    - `VehicleType`: pie chart
    - `DetectionTime_O` / `DetectionTime_D`: histogram in 5-minute bins
    - `TripLength`: numeric histogram with fine bins
    - `GantryID_O` / `GantryID_D`: bar chart of top gantry codes
  - Plots are rendered with a journal-like Matplotlib style inside the main window, with an adjacent text panel showing summary statistics

- **Spatio-temporal statistics**
  - Choose a time column and a node column
  - Specify aggregation granularity (in minutes)
  - Generate a spatio-temporal flow matrix:
    - Rows: time bins
    - Columns: nodes
    - Values: counts of records per (time_bin, node)
  - Display the matrix in the main table view (time_bin + first 10 node columns as a sample)
  - Show the full matrix shape in the status bar
  - Flow map for a single node:
    - Input a node index (1-based)
    - Plot a time-series line chart of the node’s flow over time
  - 3D spatio-temporal surface:
    - Use the full time × node matrix as a height field
    - Smooth in both time and node dimensions for a topographic-style surface plot

- **Global behavior**
  - Excel-like table styling
  - All operations and errors are logged to `logs/trafficlens.log`
  - The main window has a global “Back” button at the top-right:
    - If a plot is visible, it closes the plot and returns to the current table
    - In spatio-temporal mode, it can also restore the original traffic-record view

### Installation and usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application from the project root:

```bash
python main.py
```

3. Typical workflow:

- Open the **File** tab to import CSV files or folders
- Use the **Operations** tab to search and sort records
- Use the **View** tab to filter by field and plot descriptive statistics
- Use the **Spatio-temporal statistics** tab to generate time × node matrices, flow maps and 3D plots
- Export any current result from the **File** tab as CSV/XLSX/NPY

---

### 项目简介

TrafficLens 是一个基于 Python 的交通数据查询与分析桌面应用，  
使用 Tkinter 构建类 Excel 的图形界面，并将前端界面与后端数据逻辑清晰地解耦。

适用场景包括：

- 导入大规模交通 CSV 数据集
- 交互式检索、排序和过滤行程记录
- 进行交通基础统计分析和高质量可视化
- 按时间 × 节点聚合生成时空流量矩阵
- 导出处理结果用于后续建模与论文绘图

#### 数据格式

每一行代表一条行程（或一次检测），包含 8 个字段：

1. `VehicleType`：车辆类型（类别变量）
2. `DetectionTime_O`：起点检测时间
3. `GantryID_O`：起点门架 / 节点编码
4. `DetectionTime_D`：终点检测时间
5. `GantryID_D`：终点门架 / 节点编码
6. `TripLength`：行程长度（距离或时间等数值）
7. `TripEnd`：行程结束编码
8. `TripInformation`：行程文本信息

不同字段在统计模块中会按时间型、代码型、数值型、文本型等进行差异化处理。

#### 主要功能

- **文件**
  - 导入单个 CSV 文件
  - 导入文件夹中全部 CSV
  - 将额外文件 / 文件夹合并到当前数据集中
  - 将当前结果（原始表 / 过滤结果 / 时空矩阵）导出为 CSV、XLSX 或 NPY
  - 一键清空所有数据，重新开始分析

- **操作**
  - 关键字搜索（模糊 / 严格）
  - 全局搜索或指定列搜索
  - 显示匹配行数，并在表格中高亮匹配行
  - 依据当前选中列进行升序 / 降序排序
  - 支持分页显示（跳转页、上一页 / 下一页），适用于大规模数据

- **视图**
  - 针对以下字段进行视图过滤：
    - `DetectionTime_O`、`DetectionTime_D`：时间范围过滤
    - `VehicleType`：类别子集过滤
    - `TripLength`：数值范围过滤
  - 根据字段类型自动切换输入控件，提示合理的过滤区间
  - 同一界面中集成统计绘图功能：
    - `VehicleType`：饼图统计
    - `DetectionTime_O` / `DetectionTime_D`：5 分钟间隔的时间直方图
    - `TripLength`：高分辨率数值直方图
    - `GantryID_O` / `GantryID_D`：门架代码频数柱状图
  - 所有图像均嵌入主窗口中展示，右侧文本框显示详细统计信息，绘图风格对标顶级期刊。

- **时空统计**
  - 选择时间列与节点列
  - 输入时间粒度（单位：分钟）
  - 基于当前数据集构建时空流量矩阵：
    - 行：时间粒度划分后的时间段
    - 列：节点（门架编码）
    - 值：每个时间段 / 节点内的通过流量（记录条数）
  - 在主表中展示 `time_bin + 前 10 个节点` 的示例列，并在底部状态栏展示完整矩阵形状
  - 流量图功能：
    - 输入节点序号（从 1 开始）
    - 绘制该节点在整个时间维度上的流量时序折线图
  - 三维时空图：
    - 使用完整的 time × node 矩阵作为高度场
    - 在时间和节点两个方向上进行插值与轻微平滑，获得类似“地形图”的平滑 3D 曲面

- **整体行为**
  - 表格外观尽量贴近 Excel，列宽可拖动调整
  - 所有操作与错误统一记录在 `logs/trafficlens.log`，便于调试与溯源
  - 顶部右侧提供全局“回退”按钮：
    - 若当前有图像，则关闭图像返回表格视图
    - 若处于时空统计视图，可返回生成前的原始交通记录表

### 安装与运行方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 在项目根目录运行：

```bash
python main.py
```

3. 推荐使用流程：

- 在 **文件** 标签页中导入 CSV 数据或文件夹
- 在 **操作** 中进行搜索和排序，快速定位关心的记录
- 在 **视图** 中基于字段进行过滤，并绘制对应的统计图
- 在 **时空统计** 中构建时空流量矩阵，查看流量图与 3D 时空曲面
- 随时在 **文件** 标签页中导出当前结果（CSV / XLSX / NPY）以供后续分析和论文作图
