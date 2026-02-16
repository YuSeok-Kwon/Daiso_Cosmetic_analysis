-- Why-pi 데이터베이스 스키마
-- 11개 테이블 정규화 구조

-- ============================================
-- 마스터 테이블
-- ============================================

-- 1. 브랜드 테이블
CREATE TABLE brands (
    brand_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL
);

-- 2. 카테고리 테이블
CREATE TABLE categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_home VARCHAR(50),
    category_1 VARCHAR(50),
    category_2 VARCHAR(50),
    UNIQUE KEY uk_category (category_home, category_1, category_2)
);

-- 3. 성분 마스터 테이블
CREATE TABLE ingredients_master (
    ingredient_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    is_caution BOOLEAN DEFAULT FALSE,
    is_haram BOOLEAN DEFAULT FALSE,
    is_animal_derived BOOLEAN DEFAULT FALSE
);

-- ============================================
-- 제품 테이블
-- ============================================

-- 4. 제품 기본 정보 테이블
CREATE TABLE products (
    product_code INT PRIMARY KEY,
    brand_id INT NOT NULL,
    category_id INT NOT NULL,
    name VARCHAR(500) NOT NULL,
    price INT,
    country VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (brand_id) REFERENCES brands(brand_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id),
    INDEX idx_brand (brand_id),
    INDEX idx_category (category_id)
);

-- 5. 제품 속성 테이블 (1:1 관계)
CREATE TABLE product_attributes (
    product_code INT PRIMARY KEY,
    `group` VARCHAR(255),
    is_functional BOOLEAN DEFAULT FALSE,
    price_tier VARCHAR(50),
    price_position VARCHAR(50),
    item_ph VARCHAR(20),
    whitening BOOLEAN DEFAULT FALSE,
    wrinkle_reduction BOOLEAN DEFAULT FALSE,
    sunscreen BOOLEAN DEFAULT FALSE,
    can_vegan BOOLEAN,
    can_halal BOOLEAN,
    FOREIGN KEY (product_code) REFERENCES products(product_code) ON DELETE CASCADE
);

-- 6. 제품 지표 테이블 (1:1 관계)
CREATE TABLE product_metrics (
    product_code INT PRIMARY KEY,
    likes INT DEFAULT 0,
    shares INT DEFAULT 0,
    review_count INT DEFAULT 0,
    engagement_score DECIMAL(10, 2),
    cp_index DECIMAL(10, 2),
    is_god_sung_bi BOOLEAN DEFAULT FALSE,
    review_density DECIMAL(10, 6),
    last_updated DATE,
    FOREIGN KEY (product_code) REFERENCES products(product_code) ON DELETE CASCADE
);

-- 7. 제품-성분 연결 테이블 (N:M 관계)
CREATE TABLE product_ingredients (
    product_code INT NOT NULL,
    ingredient_id INT NOT NULL,
    `rank` INT NOT NULL,
    PRIMARY KEY (product_code, ingredient_id),
    FOREIGN KEY (product_code) REFERENCES products(product_code) ON DELETE CASCADE,
    FOREIGN KEY (ingredient_id) REFERENCES ingredients_master(ingredient_id),
    INDEX idx_ingredient (ingredient_id)
);

-- ============================================
-- 사용자/리뷰 테이블
-- ============================================

-- 8. 사용자 테이블
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    user_masked VARCHAR(50),
    activity_level VARCHAR(20),
    rating_tendency VARCHAR(50),
    INDEX idx_activity (activity_level)
);

-- 9. 리뷰 테이블
CREATE TABLE reviews (
    order_id INT PRIMARY KEY,
    product_code INT NOT NULL,
    user_id INT NOT NULL,
    write_date DATE,
    rating DECIMAL(2, 1),
    text TEXT,
    image_count INT DEFAULT 0,
    is_reorder BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (product_code) REFERENCES products(product_code),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_product (product_code),
    INDEX idx_user (user_id),
    INDEX idx_date (write_date)
);

-- 10. 리뷰 분석 테이블 (1:1 관계)
CREATE TABLE review_analysis (
    order_id INT PRIMARY KEY,
    length INT,
    length_category VARCHAR(20),
    sentiment VARCHAR(20),
    sentiment_score DECIMAL(5, 4),
    promo_type VARCHAR(50),
    FOREIGN KEY (order_id) REFERENCES reviews(order_id) ON DELETE CASCADE
);

-- ============================================
-- 프로모션 테이블
-- ============================================

-- 11. 프로모션 테이블
CREATE TABLE promotions (
    promotion_id INT AUTO_INCREMENT PRIMARY KEY,
    brand_id INT,
    start_date DATE,
    end_date DATE,
    description VARCHAR(500),
    event_type VARCHAR(100),
    FOREIGN KEY (brand_id) REFERENCES brands(brand_id),
    INDEX idx_brand (brand_id),
    INDEX idx_date (start_date, end_date)
);

-- ============================================
-- 인덱스 추가 (성능 최적화)
-- ============================================

-- 복합 인덱스
CREATE INDEX idx_reviews_product_date ON reviews(product_code, write_date);
CREATE INDEX idx_reviews_user_date ON reviews(user_id, write_date);
