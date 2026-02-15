#!/usr/bin/env python3
"""
GCS에 CSV 파일 업로드 스크립트
사용법: python upload_to_gcs.py <bucket_name>
"""

import os
import sys
from pathlib import Path


def upload_many_blobs_with_transfer_manager(
    bucket_name, filenames, source_directory="", workers=8
):
    """Upload every file in a list to a bucket, concurrently in a process pool."""

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    results = transfer_manager.upload_many_from_filenames(
        bucket, filenames, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(filenames, results):
        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))


def main():
    # 버킷 이름 설정 (명령줄 인자 또는 직접 수정)
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
    else:
        # TODO: 여기에 버킷 이름을 직접 입력하세요
        bucket_name = "your-bucket-name"
        print(f"사용법: python {sys.argv[0]} <bucket_name>")
        print(f"또는 스크립트 내 bucket_name 변수를 수정하세요.")
        sys.exit(1)

    # 소스 디렉토리
    source_directory = "/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/Data/csv/union"

    # 업로드할 파일 목록
    filenames = [
        "brands.csv",
        "categories.csv",
        "ingredients_master.csv",
        "product_attributes.csv",
        "product_ingredients.csv",
        "product_metrics.csv",
        "products.csv",
        "promotions.csv",
        "review_analysis.csv",
        "reviews.csv",
        "users.csv",
    ]

    print(f"버킷: {bucket_name}")
    print(f"소스 디렉토리: {source_directory}")
    print(f"업로드할 파일 수: {len(filenames)}")
    print("-" * 40)

    upload_many_blobs_with_transfer_manager(
        bucket_name=bucket_name,
        filenames=filenames,
        source_directory=source_directory,
        workers=8
    )

    print("-" * 40)
    print("업로드 완료!")


if __name__ == "__main__":
    main()
