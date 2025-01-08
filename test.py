# visualize.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 设置中文字体（确保系统中有支持的中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Streamlit 页面配置必须是第一个 Streamlit 命令
st.set_page_config(page_title="可转债多日期交易信号可视化", layout="wide")

st.title("可转债多日期交易信号可视化")

# 定义文件路径
output_dir = "D:/YZY/convertible_daily/intermediate_outputs"
file_path = os.path.join(output_dir, "converted_bollinger_with_trading_signals.csv")

# 读取数据
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', parse_dates=['Date'])
        return df
    except FileNotFoundError:
        st.error(f"文件未找到: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"读取文件时出错: {e}")
        return pd.DataFrame()

df = load_data(file_path)

if df.empty:
    st.stop()

# 中文列名映射
column_mapping = {
    'Bond': '债券代码',
    'Date': '日期',
    'Price': '价格',
    'PctChange': '每日涨跌幅',
    '20MA': '20日均线',
    '20STD': '20日标准差',
    'UpperBand': '上轨',
    'LowerBand': '下轨',
    'BuySignal': '买入信号',
    'SellSignal': '卖出信号',
    'SupportLevel': '支撑位',
    'ResistanceLevel': '压力位',
    'BuyAlert': '买入提醒',
    'SellAlert': '卖出提醒',
    # 交易信号相关
    'Buy_Signal_1M_10P': '买入信号_1M_10%',
    'Sell_Signal_1M_10P': '卖出信号_1M_10%',
    'Support_Operation_1M_10P': '支撑操作_1M_10%',
    'Resistance_Operation_1M_10P': '压力操作_1M_10%',
    'Buy_Signal_1M_20P': '买入信号_1M_20%',
    'Sell_Signal_1M_20P': '卖出信号_1M_20%',
    'Support_Operation_1M_20P': '支撑操作_1M_20%',
    'Resistance_Operation_1M_20P': '压力操作_1M_20%',
    'Buy_Signal_3M_10P': '买入信号_3M_10%',
    'Sell_Signal_3M_10P': '卖出信号_3M_10%',
    'Support_Operation_3M_10P': '支撑操作_3M_10%',
    'Resistance_Operation_3M_10P': '压力操作_3M_10%',
    'Buy_Signal_3M_20P': '买入信号_3M_20%',
    'Sell_Signal_3M_20P': '卖出信号_3M_20%',
    'Support_Operation_3M_20P': '支撑操作_3M_20%',
    'Resistance_Operation_3M_20P': '压力操作_3M_20%',
    'Buy_Signal_6M_10P': '买入信号_6M_10%',
    'Sell_Signal_6M_10P': '卖出信号_6M_10%',
    'Support_Operation_6M_10P': '支撑操作_6M_10%',
    'Resistance_Operation_6M_10P': '压力操作_6M_10%',
    'Buy_Signal_6M_20P': '买入信号_6M_20%',
    'Sell_Signal_6M_20P': '卖出信号_6M_20%',
    'Support_Operation_6M_20P': '支撑操作_6M_20%',
    'Resistance_Operation_6M_20P': '压力操作_6M_20%',
    # 其他可能的列名
    'Strong_Up': '强势上涨',
    'Strong_Down': '强势下跌',
    'Consec_Up_Count': '连续上涨计数',
    'Consec_Down_Count': '连续下跌计数',
    'Consec_Up_Avg_PctChange': '连续上涨平均涨幅',
    'Consec_Down_Avg_PctChange': '连续下跌平均跌幅',
    'Consec_Up_Pctile_1MD': '连续上涨百分位_1M',
    'Consec_Up_Pctile_3MD': '连续上涨百分位_3M',
    'Consec_Up_Pctile_6MD': '连续上涨百分位_6M',
    'Consec_Down_Pctile_1MD': '连续下跌百分位_1M',
    'Consec_Down_Pctile_3MD': '连续下跌百分位_3M',
    'Consec_Down_Pctile_6MD': '连续下跌百分位_6M',
    'Strong_Up_Alert_1M_10P': '强势上涨提醒_1M_10%',
    'Strong_Up_Alert_1M_20P': '强势上涨提醒_1M_20%',
    'Strong_Up_Alert_3M_10P': '强势上涨提醒_3M_10%',
    'Strong_Up_Alert_3M_20P': '强势上涨提醒_3M_20%',
    'Strong_Up_Alert_6M_10P': '强势上涨提醒_6M_10%',
    'Strong_Up_Alert_6M_20P': '强势上涨提醒_6M_20%',
    'Strong_Down_Alert_1M_10P': '强势下跌提醒_1M_10%',
    'Strong_Down_Alert_1M_20P': '强势下跌提醒_1M_20%',
    'Strong_Down_Alert_3M_10P': '强势下跌提醒_3M_10%',
    'Strong_Down_Alert_3M_20P': '强势下跌提醒_3M_20%',
    'Strong_Down_Alert_6M_10P': '强势下跌提醒_6M_10%',
    'Strong_Down_Alert_6M_20P': '强势下跌提醒_6M_20%',
    # 添加更多列名映射根据需要...
}

# 执行列名映射
df.rename(columns=column_mapping, inplace=True)

# 检查必要的列是否存在
required_columns = ['债券代码', '日期', '价格', '每日涨跌幅']
missing_required = [col for col in required_columns if col not in df.columns]
if missing_required:
    st.error(f"缺少以下必要的列: {missing_required}")
    st.stop()

# 合并交易信号列
def merge_signals(df, window, percentile):
    buy_col = f'买入信号_{window}_{percentile}%'
    sell_col = f'卖出信号_{window}_{percentile}%'
    support_col = f'支撑操作_{window}_{percentile}%'
    resistance_col = f'压力操作_{window}_{percentile}%'
    merged_col = f'交易信号_{window}_{percentile}%'

    # 检查这些列是否存在
    for col in [buy_col, sell_col, support_col, resistance_col]:
        if col not in df.columns:
            st.warning(f"列 {col} 不存在于数据中。请检查数据源和列名映射。")
            df[col] = ''

    def merge_row(row):
        signals = []
        if pd.notna(row[buy_col]) and str(row[buy_col]).strip() != '':
            signals.append(str(row[buy_col]).strip())
        if pd.notna(row[sell_col]) and str(row[sell_col]).strip() != '':
            signals.append(str(row[sell_col]).strip())
        if pd.notna(row[support_col]) and str(row[support_col]).strip() != '':
            signals.append(str(row[support_col]).strip())
        if pd.notna(row[resistance_col]) and str(row[resistance_col]).strip() != '':
            signals.append(str(row[resistance_col]).strip())
        if signals:
            return ', '.join(signals)
        else:
            return ""

    df[merged_col] = df.apply(merge_row, axis=1)
    return df

# 定义时间窗口和百分位
windows = ['1M', '3M', '6M']
percentiles = [10, 20]

# 合并所有信号
for window in windows:
    for percentile in percentiles:
        df = merge_signals(df, window, percentile)


# 定义交易信号的中文映射
signal_translation = {
    'Buy/Add_Position': '可建仓/加仓',
    'Partial/Full_Stop_Loss_Take_Profit': '部分/全部止损获利',
    'Buy/Add_Position_at_Support_with_Up_Alert': '在支撑位上加仓并有上涨警报',
    'Small_Buy/Add_Position_at_Support': '在支撑位小规模买入/加仓',
    'Partial/Full_Stop_Loss_at_Resistance_with_Down_Alert': '在压力位部分/全部止损并有下跌警报',
    'Small_Decrease_Stop_Loss_at_Resistance': '在压力位小规模减仓止损'
}

# 应用映射到所有交易信号列
signal_columns_merged = [
    '交易信号_1M_10%', '交易信号_1M_20%',
    '交易信号_3M_10%', '交易信号_3M_20%',
    '交易信号_6M_10%', '交易信号_6M_20%'
]

for col in signal_columns_merged:
    df[col] = df[col].replace(signal_translation)





# 确保交易信号列为字符串类型
signal_columns_merged = [
    '交易信号_1M_10%', '交易信号_1M_20%',
    '交易信号_3M_10%', '交易信号_3M_20%',
    '交易信号_6M_10%', '交易信号_6M_20%'
]

for col in signal_columns_merged:
    if col in df.columns:
        df[col] = df[col].astype(str)
    else:
        st.warning(f"列 {col} 不存在于数据中。请检查数据源和列名。")

# 添加调试信息：显示列名
st.sidebar.subheader("调试信息")
if st.sidebar.checkbox("显示所有列名"):
    st.sidebar.write(df.columns.tolist())

# 过滤选项
st.sidebar.header("筛选选项")

# 选择日期范围
min_date = df['日期'].min()
max_date = df['日期'].max()

selected_date_range = st.sidebar.date_input(
    "选择日期范围",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(selected_date_range) != 2:
    st.sidebar.error("请选择一个有效的日期范围。")
    st.stop()

start_date, end_date = selected_date_range
df_filtered = df[(df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))]

# 选择时间窗口和百分位
selected_window = st.sidebar.selectbox("选择时间窗口", options=windows)
selected_percentile = st.sidebar.selectbox("选择百分位", options=percentiles)

# 构建合并后的交易信号列名
selected_signal_col = f'交易信号_{selected_window}_{selected_percentile}%'

# 检查选定的列是否存在
if selected_signal_col not in df.columns:
    st.error(f"选定的交易信号列 '{selected_signal_col}' 不存在。请检查数据源和列名映射。")
    st.stop()

# 筛选有交易信号的行（在选定日期范围内）
df_signals = df_filtered[df_filtered[selected_signal_col] != ""]

# 获取所有唯一的交易信号描述
unique_signals = df_signals[selected_signal_col].dropna().unique().tolist()

# 添加交易信号结果筛选
st.sidebar.header("交易信号结果筛选")
selected_signals = st.sidebar.multiselect(
    "选择交易信号结果",
    options=unique_signals,
    default=unique_signals  # 默认全选
)

# 如果用户选择了特定的交易信号结果，则进一步筛选
if selected_signals:
    # 使用正则表达式进行匹配，确保包含任意一个选中的交易信号
    pattern = '|'.join([f'\\b{signal}\\b' for signal in selected_signals])
    df_signals = df_signals[df_signals[selected_signal_col].str.contains(pattern, regex=True, na=False)]
else:
    # 如果没有选择任何信号，则显示空
    df_signals = pd.DataFrame(columns=df.columns)

# 显示全部债券数据（在选定日期范围内）
st.subheader("可转债数据（全部，选定日期范围）")
st.dataframe(df_filtered)

# 显示筛选后的交易信号数据，仅包含特定列
st.subheader(f"筛选后的交易信号（时间窗口：{selected_window}，百分位数：{selected_percentile}%）")
if not df_signals.empty:
    # 选择需要展示的列
    columns_to_display = ['债券代码', '日期', '价格', '每日涨跌幅', selected_signal_col]
    # 检查这些列是否存在
    missing_display_columns = [col for col in columns_to_display if col not in df_signals.columns]
    if missing_display_columns:
        st.error(f"以下列不存在于数据中: {missing_display_columns}")
    else:
        df_signals_display = df_signals[columns_to_display]
        st.dataframe(df_signals_display)
else:
    st.write("当前筛选条件下，没有交易信号数据。")

# 新增债券筛选功能
st.sidebar.header("债券筛选")
bond_codes = df['债券代码'].unique().tolist()
selected_bond = st.sidebar.selectbox("选择债券代码", options=bond_codes)

# 筛选特定债券的数据（在选定日期范围内）
df_bond = df_filtered[df_filtered['债券代码'] == selected_bond]

# 显示特定债券的全部底层数据
st.subheader(f"债券代码：{selected_bond} 的全部数据（选定日期范围内）")
st.dataframe(df_bond)

# 可视化特定债券的每日涨跌幅趋势
st.subheader(f"{selected_bond} 的每日涨跌幅趋势")

fig, ax = plt.subplots(figsize=(14,7))
sns.lineplot(data=df_bond, x='日期', y='每日涨跌幅', label='每日涨跌幅', color='red', ax=ax)

plt.xlabel('日期', fontsize=12)
plt.ylabel('每日涨跌幅 (%)', fontsize=12)
plt.title(f'{selected_bond} 的每日涨跌幅趋势', fontsize=16)
plt.xticks(rotation=45)
plt.legend(fontsize=12)
st.pyplot(fig)

# 可选：展示特定债券的价格与交易信号
st.subheader(f"{selected_bond} 的价格与交易信号标记（选定日期范围内）")

fig2, ax2 = plt.subplots(figsize=(14,7))
sns.lineplot(data=df_bond, x='日期', y='价格', label='价格', color='orange', ax=ax2)

# 筛选特定债券的交易信号
df_bond_signals = df_bond[df_bond[selected_signal_col] != ""]

# 标记筛选后的交易信号
if not df_bond_signals.empty:
    sns.scatterplot(data=df_bond_signals, x='日期', y='价格', hue=selected_signal_col,
                    style=selected_signal_col, s=200, ax=ax2, palette='coolwarm', edgecolor='white')
    plt.legend(title='交易信号', fontsize=12, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
else:
    st.write("当前筛选条件下，没有交易信号标记。")

plt.xlabel('日期', fontsize=12)
plt.ylabel('价格', fontsize=12)
plt.title(f'{selected_bond} 的价格与交易信号', fontsize=16)
plt.xticks(rotation=45)
plt.legend(fontsize=12)
st.pyplot(fig2)

# 导出筛选后的数据
st.sidebar.subheader("导出选项")
export = st.sidebar.checkbox("导出筛选后的数据为 CSV")

if export and not df_signals.empty:
    # 定义导出文件名，包含时间窗口、百分位数和选择的交易信号
    selected_signals_str = "_".join(selected_signals).replace("/", "_").replace(", ", "_")
    export_filename = f'exported_results_{selected_window}_{selected_percentile}P_{selected_signals_str}.csv'
    export_path = os.path.join(output_dir, export_filename)
    df_signals.to_csv(export_path, index=False, encoding='utf-8-sig')
    st.sidebar.success(f"数据已导出至: {export_path}")
elif export and df_signals.empty:
    st.sidebar.warning("当前筛选条件下，没有数据可导出。")
