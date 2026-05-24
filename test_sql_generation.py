#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test SQL generation on the problematic query"""
import sys
sys.path.insert(0, '.')

from sql_generator import SQLGenerator

generator = SQLGenerator()

test_query = "公司的engineering部门有多少员工"

result = generator.generate_sql(test_query)
print(f"Query: {test_query}")
print(f"Success: {result.get('success')}")
print(f"SQL: {result.get('sql')}")
print(f"Result: {result.get('result')}")
print(f"Answer: {result.get('answer')}")
print(f"Error: {result.get('error')}")
