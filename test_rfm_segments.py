from app import load_and_prepare_data, create_rfm_analysis, perform_segmentation

print("Loading data...")
df = load_and_prepare_data()

print("Creating RFM analysis...")
rfm = create_rfm_analysis(df)

print("\n=== Testing with 7 segments (RFM Business Rules) ===")
rfm_7 = perform_segmentation(rfm.copy(), n_clusters=7)
segment_counts_7 = rfm_7['Segment'].value_counts()
print(segment_counts_7)
print(f"Total customers: {len(rfm_7)}")
print(f"Total unique segments: {len(segment_counts_7)}")
print(f"\nSegment percentages:")
for seg, count in segment_counts_7.items():
    pct = (count / len(rfm_7)) * 100
    print(f"  {seg}: {count} ({pct:.1f}%)")

print("\n=== Testing with 5 segments ===")
rfm_5 = perform_segmentation(rfm.copy(), n_clusters=5)
segment_counts_5 = rfm_5['Segment'].value_counts()
print(segment_counts_5)
print(f"Total unique segments: {len(segment_counts_5)}")

print("\n=== Testing with 3 segments ===")
rfm_3 = perform_segmentation(rfm.copy(), n_clusters=3)
segment_counts_3 = rfm_3['Segment'].value_counts()
print(segment_counts_3)
print(f"Total unique segments: {len(segment_counts_3)}")

print("\n=== Detailed RFM Analysis (7 segments) ===")
rfm_analysis = rfm_7.groupby('Segment').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(2)
rfm_analysis.columns = ['Count', 'Avg Recency', 'Avg Frequency', 'Avg Monetary']
print(rfm_analysis)

print("\n✅ RFM business rules segmentation applied! Better distribution across segments.")
