/*
othello.c
オセロです。最初は白からで交互に黒と打っていく。パスの回数は３回までで間違えたところに打てばパスとされる。4回パスすれば負けとなる。
自分が打ちたいところにマウスをクリックして、石を打っていく。下の余白に今の白と黒の数、パスの回数が書かれる。また勝負が終われば
下の余白にどちらが勝ったのかを描く。工夫した点は配列の場所とマウスでクリックした位置と一致させるために変数を使って表現をしたところです。
また打てるかどうかの判定とひっくり返すための処理を書くために条件分岐と繰り返しを使いながら表現をしたところです。

Hiroki Kurokawa
*/

#include <handy.h>
#include <stdio.h>

#define WINDOWSIZE 400

int main() {
    hgevent *event;
    hgsound othelloSound;
    othelloSound = HgSoundLoad("othello.mp3");  //音を出すために読み込む
    int mouse_x, mouse_y;                       //マウスの位置
    int grid[10][10] = {};                      //オセロの碁盤
    int count = 0;  //置けるかどうかの判定の変数
    int a, b;  //二次元配列とマウス座標を一致させるための変換変数
    int judge;             //白の番か黒の番を判定するための変数
    int num_1, num_2;      //白の数と黒の数を数えるための変数
    int pass_1, pass_2;    //パスした回数を覚えるための変数
    int turn_i, turn_j;    //ひっくり返すための変数
    int check_i, check_j;  //置けるかどうかを調べるための変数

    //下に余白をつけるためにy座標に+50する
    HgOpen(WINDOWSIZE, WINDOWSIZE + 50);

    HgSetEventMask(HG_MOUSE_DOWN);

    //最初の碁の初期値
    HgCircleFill(175, 275, 25, 1);
    HgCircleFill(225, 225, 25, 1);
    HgSetFillColor(HG_BLACK);
    HgCircleFill(175, 225, 25, 1);
    HgCircleFill(225, 275, 25, 1);
    //配列に入れる。１の時は白、−１の時は黒、0の時は何も書かない。
    grid[4][4] = 1;
    grid[5][4] = -1;
    grid[5][5] = 1;
    grid[4][5] = -1;

    //白から先手にする
    judge = -1;
    for (;;) {
        //五番を作るための線
        for (int i = 1; i < 9; i++) {
            HgLine(0, i * WINDOWSIZE / 8, WINDOWSIZE, i * WINDOWSIZE / 8);
            HgLine(i * WINDOWSIZE / 8, 50, i * WINDOWSIZE / 8, WINDOWSIZE + 50);
        }
        // jugdeの値によってテキストを出力する
        if (judge == -1) HgText(200, 25, "白の番です");
        if (judge == 1) HgText(200, 25, "黒の番です");
        event = HgEvent();
        //マウスの反応があってから画面を一度消す
        HgClear();
        mouse_x = event->x;
        mouse_y = event->y;

        for (int i = 1; i <= 8; i++) {
            for (int j = 1; j <= 8; j++) {
                //マウスが区切り内にあってかつ押したところが何も置いていながったら
                if ((i * 50 <= mouse_x + 50 && mouse_x + 50 < (i + 1) * 50) &&
                    (j * 50 <= mouse_y && mouse_y < (j + 1) * 50) &&
                    grid[i][9 - j] == 0) {
                    //配列の値と一致させるために変換させる
                    a = i;
                    b = 9 - j;
                    //もし今が白ならば次の番を黒にしてループから抜ける
                    if (judge == -1) {
                        judge = 1;
                        break;
                    }
                    //もし黒ならば次の番を白にしてループから抜ける
                    if (judge == 1) {
                        judge = -1;
                        break;
                    }
                }
            }
        }
        //８方向に同じ色の石があるかどうかを確認する
        for (int di = -1; di < 2; di++) {
            for (int dj = -1; dj < 2; dj++) {
                //値が配列からでないように条件をつける
                if (a + di >= 1 && a + di <= 8 && b + dj >= 1 && b + dj <= 8) {
                    //初期値
                    check_i = a + di;
                    check_j = b + dj;
                    //もし相手のマスならばループさせる
                    while (grid[check_i][check_j] == -judge) {
                        check_i += di;
                        check_j += dj;
                        //もし同じ色に当たったら打てると判定して音を鳴らし、ループから抜ける
                        if (grid[check_i][check_j] == judge) {
                            grid[a][b] = judge;
                            count++;
                            HgSoundPlay(othelloSound);
                            break;
                        }
                    }
                }
                if (count == 1) break;
            }
            if (count == 1) break;
        }
        //もし打てなければパスとして扱う
        if (count != 1) {
            if (judge == 1) {
                pass_1++;
            }
            if (judge == -1) {
                pass_2++;
            }
        }
        count = 0;

        //石をひっくり返すためのループ
        for (int di = -1; di < 2; di++) {
            for (int dj = -1; dj < 2; dj++) {
                //配列から出ないための条件
                if (a + di >= 1 && a + di <= 8 && b + dj >= 1 && b + dj <= 8) {
                    //初期値
                    turn_i = a + di;
                    turn_j = b + dj;
                    //相手の石にあたるならループさせる
                    while (grid[turn_i][turn_j] == -judge) {
                        //値を足す
                        turn_i += di;
                        turn_j += dj;
                        //もし同じ色と当たれば
                        if (grid[turn_i][turn_j] == judge) {
                            //下がっていく
                            turn_i -= di;
                            turn_j -= dj;
                            //打った石のところまで下がりつつ、今の色と同じ値を入れていく
                            while (!(a == turn_i && b == turn_j)) {
                                grid[turn_i][turn_j] = judge;
                                turn_i -= di;
                                turn_j -= dj;
                            }
                        }
                    }
                }
            }
        }
        //石を描くためのループ
        for (int i = 1; i <= 8; i++) {
            for (int j = 1; j <= 8; j++) {
                //もし何か値が入っていれば
                if (grid[i][j] != 0) {
                    //もし１ならば白にして白の数を増やす
                    if (grid[i][j] == 1) {
                        HgSetFillColor(HG_WHITE);
                        num_1++;
                    }
                    //もし−１なら黒にして黒の数を増やす
                    if (grid[i][j] == -1) {
                        HgSetFillColor(HG_BLACK);
                        num_2++;
                    }
                    //円を描く
                    HgCircleFill(i * WINDOWSIZE / 8 + 25 - 50,
                                 (9 - j) * WINDOWSIZE / 8 + 25, 25, 1);
                }
            }
        }
        //余白に文字を描く
        HgSetFont(HG_M, 13);
        HgText(0, 25, "白の個数:%d 黒の個数: %d ", num_1, num_2);
        HgText(0, 5,
               "白のパス回数:%d 黒のパス回数:%d (パスは３回までできます。）",
               pass_1, pass_2);
        //もし全部で６４個もしくはパスの回数が４回になれば勝者をテキストで書く
        if (num_1 + num_2 == 64 || pass_1 == 4 || pass_2 == 4) {
            if (num_1 > num_2 || pass_2 == 4) {
                HgText(200, 25, "白の勝ちです");
                break;
            }
            if (num_1 == num_2 && pass_1 != 4 && pass_2 != 4) {
                HgText(200, 25, "引き分けです");
                break;
            }
            if (num_1 < num_2 || pass_1 == 4) {
                HgText(200, 25, "黒の勝ちです");
                break;
            }
        }
        //石の数の初期化
        num_1 = 0;
        num_2 = 0;
    }
    HgGetChar();
    HgClose();
    return 0;
}