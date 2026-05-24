#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test query router on the problematic query"""
import sys
sys.path.insert(0, '.')

from query_router import QueryRouter

router = QueryRouter()

# 测试用户的查询
test_queries = [
    "我刚入职 第 365 天（满 1 年），能申请宠物丧假吗？需要什么材料？",
    "公司的engineering部门有多少员工",
    "什么是机器学习",
    "糖尿病注意事项"
]

for test_query in test_queries:
    print(f"\n{'='*60}")
    print(f"Testing: {test_query}")
    print(f"{'='*60}")
    result = router.classify_query(test_query)
    print(f"Query Type: {result.query_type}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Intent: {result.intent}")
