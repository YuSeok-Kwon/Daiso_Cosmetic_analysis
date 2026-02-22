"""
시각화 모듈

분석 결과를 시각화하는 함수들을 제공합니다.
"""

from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from collections import Counter


# 한글 폰트 설정
import matplotlib.font_manager as fm
font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
try:
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
except:
    plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False


def create_wordcloud(
    tokens: List[str],
    title: str = '워드클라우드',
    figsize: tuple = (12, 8),
    max_words: int = 100,
    background_color: str = 'white',
    colormap: str = 'viridis',
    save_path: Optional[str] = None
) -> None:
    """
    토큰 리스트로 워드클라우드를 생성합니다.
    
    Parameters:
    -----------
    tokens : List[str]
        토큰 리스트
    title : str
        제목
    figsize : tuple
        그래프 크기
    max_words : int
        최대 표시 단어 수
    background_color : str
        배경색
    colormap : str
        색상 맵
    save_path : Optional[str]
        저장 경로 (None이면 저장 안함)
    """
    if not tokens:
        print("토큰이 비어있습니다.")
        return
    
    # 빈도 계산
    word_freq = Counter(tokens)
    
    # 워드클라우드 생성
    wordcloud = WordCloud(
        font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc',
        width=800,
        height=600,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        relative_scaling=0.5
    ).generate_from_frequencies(word_freq)
    
    # 시각화
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"워드클라우드 저장: {save_path}")
    
    plt.show()


def plot_keyword_comparison(
    df_comparison: pd.DataFrame,
    group1_col: str,
    group2_col: str,
    keyword_col: str = 'keyword',
    title: str = '키워드 빈도 비교',
    figsize: tuple = (12, 10),
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    두 그룹 간 키워드 빈도를 비교하는 막대 그래프를 생성합니다.
    
    Parameters:
    -----------
    df_comparison : pd.DataFrame
        비교 데이터프레임
    group1_col : str
        그룹 1 빈도 컬럼명
    group2_col : str
        그룹 2 빈도 컬럼명
    keyword_col : str
        키워드 컬럼명
    title : str
        제목
    figsize : tuple
        그래프 크기
    top_n : int
        표시할 상위 키워드 수
    save_path : Optional[str]
        저장 경로
    """
    # 상위 N개 추출
    df_plot = df_comparison.head(top_n).copy()
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(df_plot))
    width = 0.35
    
    # 막대 그래프
    ax.barh([i - width/2 for i in x], df_plot[group1_col], width,
            label=group1_col, color='steelblue', edgecolor='black')
    ax.barh([i + width/2 for i in x], df_plot[group2_col], width,
            label=group2_col, color='coral', edgecolor='black')
    
    # 레이블 설정
    ax.set_yticks(x)
    ax.set_yticklabels(df_plot[keyword_col])
    ax.set_xlabel('빈도', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    
    plt.show()


def plot_keyword_heatmap(
    df_matrix: pd.DataFrame,
    title: str = '키워드 카테고리 히트맵',
    figsize: tuple = (10, 8),
    cmap: str = 'YlOrRd',
    save_path: Optional[str] = None
) -> None:
    """
    키워드 카테고리별 빈도 히트맵을 생성합니다.
    
    Parameters:
    -----------
    df_matrix : pd.DataFrame
        히트맵용 데이터프레임 (행: 그룹, 열: 카테고리)
    title : str
        제목
    figsize : tuple
        그래프 크기
    cmap : str
        색상 맵
    save_path : Optional[str]
        저장 경로
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        df_matrix,
        annot=True,
        fmt='.0f',
        cmap=cmap,
        linewidths=1,
        cbar_kws={'label': '빈도'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('카테고리', fontsize=12)
    ax.set_ylabel('그룹', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"히트맵 저장: {save_path}")
    
    plt.show()


def plot_category_comparison(
    df_category: pd.DataFrame,
    category_col: str = 'category',
    freq_col: str = 'frequency',
    title: str = '카테고리별 키워드 빈도',
    figsize: tuple = (12, 6),
    color: str = 'steelblue',
    save_path: Optional[str] = None
) -> None:
    """
    카테고리별 키워드 빈도 막대 그래프를 생성합니다.
    
    Parameters:
    -----------
    df_category : pd.DataFrame
        카테고리 빈도 데이터프레임
    category_col : str
        카테고리 컬럼명
    freq_col : str
        빈도 컬럼명
    title : str
        제목
    figsize : tuple
        그래프 크기
    color : str
        막대 색상
    save_path : Optional[str]
        저장 경로
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(
        df_category[category_col],
        df_category[freq_col],
        color=color,
        edgecolor='black',
        alpha=0.8
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('카테고리', fontsize=12)
    ax.set_ylabel('빈도', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # 막대 위에 값 표시
    for i, (cat, freq) in enumerate(zip(df_category[category_col], df_category[freq_col])):
        ax.text(i, freq, f'{freq:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    
    plt.show()


def plot_grouped_bar(
    df_data: pd.DataFrame,
    groups: List[str],
    categories: List[str],
    title: str = '그룹별 카테고리 비교',
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
) -> None:
    """
    여러 그룹의 카테고리별 빈도를 비교하는 그룹 막대 그래프를 생성합니다.
    
    Parameters:
    -----------
    df_data : pd.DataFrame
        데이터프레임 (행: 그룹, 열: 카테고리)
    groups : List[str]
        그룹 이름 리스트
    categories : List[str]
        카테고리 이름 리스트
    title : str
        제목
    figsize : tuple
        그래프 크기
    save_path : Optional[str]
        저장 경로
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = range(len(categories))
    width = 0.35
    colors = ['steelblue', 'coral', 'lightgreen', 'mediumpurple']
    
    for i, group in enumerate(groups):
        offset = width * (i - len(groups) / 2 + 0.5)
        values = [df_data.loc[group, cat] if cat in df_data.columns else 0 for cat in categories]
        ax.bar([pos + offset for pos in x], values, width,
               label=group, color=colors[i % len(colors)], edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('빈도', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    
    plt.show()


def plot_rating_distribution(
    df_reviews: pd.DataFrame,
    rating_col: str = 'rating',
    group_col: Optional[str] = None,
    title: str = '평점 분포',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    평점 분포 히스토그램을 생성합니다.
    
    Parameters:
    -----------
    df_reviews : pd.DataFrame
        리뷰 데이터프레임
    rating_col : str
        평점 컬럼명
    group_col : Optional[str]
        그룹별로 나눌 컬럼명 (None이면 전체)
    title : str
        제목
    figsize : tuple
        그래프 크기
    save_path : Optional[str]
        저장 경로
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if group_col:
        # 그룹별 분포
        groups = df_reviews[group_col].unique()
        colors = ['steelblue', 'coral']
        
        for i, group in enumerate(groups):
            data = df_reviews[df_reviews[group_col] == group][rating_col]
            ax.hist(data, bins=range(1, 7), alpha=0.6, label=str(group),
                   color=colors[i % len(colors)], edgecolor='black')
        
        ax.legend()
    else:
        # 전체 분포
        ax.hist(df_reviews[rating_col], bins=range(1, 7),
               color='steelblue', edgecolor='black', alpha=0.8)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('평점', fontsize=12)
    ax.set_ylabel('리뷰 수', fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    
    plt.show()


def plot_keyword_frequency(
    df_freq: pd.DataFrame,
    keyword_col: str = 'keyword',
    freq_col: str = 'frequency',
    title: str = '키워드 빈도',
    figsize: tuple = (12, 8),
    top_n: int = 20,
    color: str = 'steelblue',
    horizontal: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    키워드 빈도 막대 그래프를 생성합니다.
    
    Parameters:
    -----------
    df_freq : pd.DataFrame
        키워드 빈도 데이터프레임
    keyword_col : str
        키워드 컬럼명
    freq_col : str
        빈도 컬럼명
    title : str
        제목
    figsize : tuple
        그래프 크기
    top_n : int
        표시할 상위 키워드 수
    color : str
        막대 색상
    horizontal : bool
        가로 막대 여부
    save_path : Optional[str]
        저장 경로
    """
    # 상위 N개 추출
    df_plot = df_freq.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizontal:
        ax.barh(df_plot[keyword_col], df_plot[freq_col],
               color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel('빈도', fontsize=12)
        ax.invert_yaxis()
    else:
        ax.bar(df_plot[keyword_col], df_plot[freq_col],
              color=color, edgecolor='black', alpha=0.8)
        ax.set_ylabel('빈도', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    
    plt.show()


def create_comparison_wordclouds(
    tokens_list1: List[str],
    tokens_list2: List[str],
    title1: str = '그룹 1',
    title2: str = '그룹 2',
    figsize: tuple = (20, 8),
    save_path: Optional[str] = None
) -> None:
    """
    두 그룹의 워드클라우드를 나란히 생성합니다.
    
    Parameters:
    -----------
    tokens_list1 : List[str]
        그룹 1 토큰 리스트
    tokens_list2 : List[str]
        그룹 2 토큰 리스트
    title1 : str
        그룹 1 제목
    title2 : str
        그룹 2 제목
    figsize : tuple
        전체 그래프 크기
    save_path : Optional[str]
        저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 그룹 1 워드클라우드
    if tokens_list1:
        word_freq1 = Counter(tokens_list1)
        wc1 = WordCloud(
            font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc',
            width=800,
            height=600,
            max_words=100,
            background_color='white',
            colormap='Blues'
        ).generate_from_frequencies(word_freq1)
        
        axes[0].imshow(wc1, interpolation='bilinear')
        axes[0].set_title(title1, fontsize=14, fontweight='bold', pad=20)
        axes[0].axis('off')
    
    # 그룹 2 워드클라우드
    if tokens_list2:
        word_freq2 = Counter(tokens_list2)
        wc2 = WordCloud(
            font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc',
            width=800,
            height=600,
            max_words=100,
            background_color='white',
            colormap='Reds'
        ).generate_from_frequencies(word_freq2)
        
        axes[1].imshow(wc2, interpolation='bilinear')
        axes[1].set_title(title2, fontsize=14, fontweight='bold', pad=20)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"워드클라우드 비교 저장: {save_path}")
    
    plt.show()
