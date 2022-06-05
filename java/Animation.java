/*
Animation.java
Hiroki Kurokawa
5/25
ランダムな文字が落ちるアニメーション
*/

import java.awt.Color;
import java.util.ArrayList;
import java.util.Random;

public class Animation {
    void run() throws InterruptedException {
        EZ.initialize(400, 400);
        this.fallText();
    }

    void fallText() throws InterruptedException {
        Random random = new Random();
        ArrayList<Integer> textY = new ArrayList<>();
        ArrayList<Character> text = new ArrayList<>();
        ArrayList<Integer> dy = new ArrayList<>();
        for (Integer i = 0; i < 26; i++) {
            textY.add(400);
            text.add('a');
            dy.add(0);
        }
        while (true) {
            for (Integer i = 0; i < 26; i++) {
                if (textY.get(i) >= 400) {
                    textY.set(i, 0);
                    text.set(i, createText());
                    dy.set(i, random.nextInt(50) + 10);
                }
            }
            for (Integer i = 0; i < 26; i++) {
                textY.set(i, textY.get(i) + dy.get(i));
                EZText text2 = EZ.addText(400 / 26 * (i + 1), textY.get(i), String.valueOf(text.get(i)), Color.BLACK);
            }
            Thread.sleep(100);
            EZ.removeAllEZElements();
        }

    }

    Character createText() {
        Random random = new Random();
        Character randomText = Character.valueOf((char) (random.nextInt(26) + 'a'));
        return randomText;
    }

    public static void main(String[] args) throws InterruptedException {
        Animation app = new Animation();
        app.run();
    }
}
