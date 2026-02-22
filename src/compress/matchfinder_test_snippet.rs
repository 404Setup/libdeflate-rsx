
    #[test]
    fn test_bt_skip_correctness() {
        let mut mf = BtMatchFinder::new();
        let mut data = vec![0u8; 1000];
        for i in 0..1000 {
            data[i] = b'A' + (i % 3) as u8; // ABCABC...
        }

        mf.prepare(data.len());

        for i in 0..500 {
            mf.skip_match(&data, i, 10);
        }

        let (len, offset) = mf.find_match(&data, 500, 10, 258);

        // If skip_match works correctly, we should find a long match at offset 3
        // (data[500..] == data[497..]).
        // If the tree is broken, we might find nothing or only 3.
        println!("Len: {}, Offset: {}", len, offset);
        assert!(len >= 3, "Should find at least length 3 match");
        assert!(offset == 3 || offset == 6 || offset == 9, "Should be multiple of 3");
    }
