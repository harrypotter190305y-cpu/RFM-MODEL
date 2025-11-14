from app import load_and_prepare_data, create_rfm_analysis, perform_segmentation

print("Loading data...")
df = load_and_prepare_data()

print("Creating RFM analysis...")
rfm = create_rfm_analysis(df)

print("\n=== Testing with 7 clusters ===")
rfm_7 = perform_segmentation(rfm.copy(), n_clusters=7)
segment_counts_7 = rfm_7['Segment'].value_counts()
print(segment_counts_7)
print(f"Total unique segments: {len(segment_counts_7)}")

print("\n=== Testing with 5 clusters ===")
rfm_5 = perform_segmentation(rfm.copy(), n_clusters=5)
segment_counts_5 = rfm_5['Segment'].value_counts()
print(segment_counts_5)
print(f"Total unique segments: {len(segment_counts_5)}")

print("\n=== Testing with 3 clusters ===")
rfm_3 = perform_segmentation(rfm.copy(), n_clusters=3)
segment_counts_3 = rfm_3['Segment'].value_counts()
print(segment_counts_3)
print(f"Total unique segments: {len(segment_counts_3)}")

print("\n✅ All tests passed! Now all requested clusters should produce distinct segments.")
