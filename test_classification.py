#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test intent classifier on the problematic query"""
import sys
sys.path.insert(0, '.')

from intent_classifier import IntentClassifier

classifier = IntentClassifier(model_path="./intent_model")

test_queries = [
    "公司的engineering部门有多少员工",
    "有多少个员工",
    "engineering部门有多少人",
    "查询engineering部门员工数量",
]

for query in test_queries:
    try:
        intent = classifier.predict(query)
        print(f"Query: {query}")
        print(f"Intent: {intent}")
        print()
    except Exception as e:
        print(f"Query: {query}")
        print(f"Error: {e}")
        print()
