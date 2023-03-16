        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # 物体を検出する
        objects = detect_objects(image)
        # 検出された物体がある場合は、レシピを表示する
        if len(objects) > 0:
            st.write('以下の食材が検出されました:')
            for obj in objects:
                st.write('- ' + obj['label'])
            st.write('以下のレシピをおすすめします:')
            # TODO: 食材に合わせたレシピを検索して表示する
        else:
            st.write('食材が検出されませんでした。もう一度画像をアップロードしてください。')
    else:
        st.write('画像をアップロードしてください。')
        
if __name__ == '__main__':
    main()
