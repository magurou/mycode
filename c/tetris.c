/*
tetris.c
テトリスです。ミノを積んで横が揃ったら消えて、スコアが増える、といったゲームです。
スコアは消したライン*100で計算されます。もし、ミノを生成するところにミノがあれば
ゲームオーバーとなり、もう一度するかどうかを表示します。そこで６０秒経てば画面を閉
じて終了します。コンパイルをして実行するとゲームが始まります。最初にミノが現れて右
に動くには'd',左に動かすには'a'、早く落とすには's'、回転をするには'c'を押せばそ
れぞれ動作します。またゲームオーバーの際に'c'を押せばもう一度ゲームをすることがで
きます。
Hiroki Kurokawa
*/
#include <handy.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WINDOWSIZE 300  //画面サイズ
#define TRUE 1  
#define FALSE -1
#define BLOCK 1 //ブロックがある
#define NONBLOCK 0 //ブロックがない
int grid[10][21] = {};  //マス目

//構造体の宣言
typedef struct Mino {
    int x;                //ミノのx座標
    int y;                //ミノのy座標
    int dy;               //ミノの落下速度
    int shape[4][4];      //ミノの形を保存する変数
    int rot_shape[4][4];  //ミノの回転したときの形を保存する変数
    int gameOver;         //ゲームオーバーの際に使う変数
} Mino;

//各関数のプロトタイプ宣言
Mino minoSetup();                 //ミノの初期設定を行う
Mino minoRotateSetup(Mino mino);  //ミノの回転処理の際に使う
Mino down_mino(Mino mino);        //ミノを落とす処理を行う
Mino move_left(Mino mino);        //ミノを左に動かす処理をする
Mino move_right(Mino mino);       //ミノを右に動かす処理をする
Mino rotate_right(Mino mino);     //ミノを右回転させる処理をする
int line_erase(Mino mino);  //ラインが揃ったかどうかを判定し、揃っているなら消す関数
int top_mino(Mino mino);    //一番上のミノの座標を返す関数
int under_mino(Mino mino);  //一番下のミノの座標を返す関数
int left_mino(Mino mino);   //一番左のミノの座標を返す関数
int right_mino(Mino mino);  //一番右のミノの座標を返す関数
int can_Down(Mino mino);   //下に落ちれるかどうかを判定する関数
int can_Left(Mino mino);   //左にいけるかどうかを判定する関数
int can_right(Mino mino);  //右に行けるかどうかを判定する関数
int can_right_rotate(Mino mino);  //右回転ができるかどうかを判定する関数
void field_color(void);  //フィールドを描く関数
int gameOver(void);      //ゲームオーバーの際に呼び出す関数
void score_color(int erase_count);  //スコアを描くための関数

int main() {
    hgevent *event;
    Mino mino;
    int first = TRUE;        //一番最初だけ起こすための変数
    int flag;             //ゲームオーバーの際に使うフラグ
    int erase_count = 0;  //消去したラインを数える変数
    srandom((unsigned int)time(NULL));

    HgOpen(WINDOWSIZE, WINDOWSIZE * 2);
    //一番下に降りれないように１を設定
    for (int i = 0; i < 10; i++) {
        grid[i][0] = BLOCK;
    }
    /*
    for(int i=0;i<10;i++){
        for(int j=0;j<21;j++){
            printf("grid[%d][%d] == %d\n", i, j, grid[i][j]);
        }
    }*/
    HgSetEventMask(HG_KEY_DOWN);
    HgSetIntervalTimer(0.5);  //ミノが落ちるまでの秒数
    for (;;) {
        // keyとintervalのイベントをとる
        event = HgEventNonBlocking();
        //最初のミノの設定
        //最初かもしくはフラグが立っているかもしくは降りれなかったらミノを初期化する
        if (first == TRUE || flag == TRUE || can_Down(mino) == FALSE) {
            //フラグをfalseにする
            flag = FALSE;
            //ラインが消せるなら消し、スコアを数える
            // printf("0ok\n");
            erase_count += line_erase(mino);
            //スコアを更新する
            score_color(erase_count);
            // printf("1ok\n");
            //最初のミノの設定をする
            mino = minoSetup();
            //ゲームオーバーの際の処理
            if (mino.gameOver == TRUE) {
                //少しだけ待機
                HgSleep(0.2);
                //ゲームオーバーしたときに続けるかどうかを関数を呼び出して判断
                flag = gameOver();
                //もし続けないなら閉じる
                if (flag == FALSE) {
                    HgClose();
                    return 0;
                }
                //続けるならミノを全て消して、設定する
                if (flag == TRUE) {
                    for (int i = 0; i < 10; i++) {
                        for (int j = 1; j < 21; j++) {
                            grid[i][j] = NONBLOCK;
                        }
                    }
                    HgClear();
                    HgSetIntervalTimer(0.5);
                    erase_count = 0;
                }
            }
        }
        // printf("description is ok\n");
        //左右に動かすときの判定
        //左の判定
        //もしaが押されてかつ左に動かせるなら左に動かす処理をする
        if (event != NULL && event->ch == 'a' && can_Left(mino) == TRUE) {
            // printf("can_left\n");
            //左に動かす
            mino = move_left(mino);
            // printf("mino.x = %d\n", mino.x);
        }
        //右の動かす処理
        //もしdを押してかつ右に動かせるなら動かす処理をする
        if (event != NULL && event->ch == 'd' && can_right(mino) == TRUE) {
            // printf("can_right\n");
            //右に動かす
            mino = move_right(mino);
        }
        //早く落とすときの判定
        if (event != NULL && event->ch == 's' && can_Down(mino) == TRUE) {
            //下に落とす
            mino = down_mino(mino);
        }
        //回転の処理
        //右回転の処理: もしcが押されてかつ回転できるなら回転処理を行う
        if (event != NULL && event->ch == 'c' && can_right_rotate(mino) == TRUE) {
            // printf("right_rotate\n");
            //回転の際の初期設定をする
            mino = minoRotateSetup(mino);
            //回転の処理をする
            mino = rotate_right(mino);
        }
        //落ちていく判定
        // printf("x = %d, y = %d\n", x, y);
        //一番最初かもしくは一定の時間が経ったら処理をする
        if (event != NULL && event->type == HG_TIMER_FIRE || first == TRUE) {
            // printf("Interval\n");
            //降りれると判定できるなら
            if (can_Down(mino) == TRUE) {
                // printf("canDown\n");
                //降りる処理をする
                mino = down_mino(mino);
                // printf("down mino\n");
            }
        }
        //フィールドを描く
        field_color();
        //一回処理をしたなら-1にしてループする
        first = FALSE;
        // printf("main's y = %d\n", y);
    }

    HgGetChar();
    HgClose();
    return 0;
}

//ミノの形を決め、グリッドにミノを入れる
Mino minoSetup() {
    Mino mino;
    int random_piece;
    int top;
    //形を保存する配列を初期化する
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mino.shape[i][j] = NONBLOCK;
        }
    }
    random_piece = random() % 6;
    // printf("rand = %d\n", random);
    switch (random_piece) {
        case 0:  // jミノ
            mino.shape[1][1] = BLOCK;
            mino.shape[2][1] = BLOCK;
            mino.shape[2][2] = BLOCK;
            mino.shape[2][3] = BLOCK;
            break;
        case 1:  // Lミノ
            mino.shape[1][2] = BLOCK;
            mino.shape[2][2] = BLOCK;
            mino.shape[3][2] = BLOCK;
            mino.shape[3][3] = BLOCK;
            break;
        case 2:  // oミノ
            mino.shape[1][1] = BLOCK;
            mino.shape[1][2] = BLOCK;
            mino.shape[2][1] = BLOCK;
            mino.shape[2][2] = BLOCK;
            break;
        case 3:  // Iミノ
            mino.shape[0][3] = BLOCK;
            mino.shape[0][2] = BLOCK;
            mino.shape[0][1] = BLOCK;
            mino.shape[0][0] = BLOCK;
            break;
        case 4:  // Sミノ
            mino.shape[0][1] = BLOCK;
            mino.shape[1][1] = BLOCK;
            mino.shape[1][2] = BLOCK;
            mino.shape[2][2] = BLOCK;
            break;
        case 5:  // Zミノ
            mino.shape[0][2] = BLOCK;
            mino.shape[1][2] = BLOCK;
            mino.shape[1][1] = BLOCK;
            mino.shape[2][1] = BLOCK;
            break;
        default:  // Tミノ
            mino.shape[0][1] = BLOCK;
            mino.shape[1][1] = BLOCK;
            mino.shape[2][1] = BLOCK;
            mino.shape[1][2] = BLOCK;
            break;
    }
    //一番上のグリッドに描くための変数
    top = 3;
    top = top - top_mino(mino);
    // y座標の初期化
    mino.y = 16 + top;
    // x座標を初期化する
    mino.x = 3;
    //落ちる速度
    mino.dy = -1;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mino.rot_shape[i][j] = NONBLOCK;
        }
    }
    //ゲームオーバーかどうかを判定する
    mino.gameOver = FALSE;
    //もしミノを新たに置くところにミノがあればゲームオーバーとする
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (mino.shape[i][j] == BLOCK && grid[mino.x + i][mino.y + j] == BLOCK) {
                mino.gameOver = TRUE;
            }
        }
    }
    //もし置けるなら置いてゲームオーバーではないとする
    if (mino.gameOver == FALSE) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (mino.shape[i][j] == BLOCK &&
                    grid[mino.x + i][mino.y + j] == NONBLOCK){
                    grid[mino.x + i][mino.y + j] = mino.shape[i][j];
                    // printf("i %d j %d\n",i,j);
                }
            }
        }
    }
    return mino;
}

//ミノの一番上の座標を返す関数
int top_mino(Mino mino) {
    int top = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            //もし一番上と仮定しているtopより大きなところで座標があれば
            if (mino.shape[i][j] == BLOCK && top < j) {
                //トップに入れる
                top = j;
            }
        }
    }
    return top;
}
//ミノごとに一番下の座標が違うからそれを調べる関数
int under_mino(Mino mino) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (mino.shape[j][i] == BLOCK) {
                //下から調べて最初に出たy座標を返す
                // printf("under_mino is %d\n", i);
                return i;
            }
        }
    }
    return -1;
}
//ミノにある一番左の座標を返す関数
int left_mino(Mino mino) {
    int left = 3;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (mino.shape[j][i] == BLOCK && left > j) {
                left = j;
            }
        }
    }
    //一番左を返す
    return left;
}
//一番右側にあるミノを返す関数
int right_mino(Mino mino) {
    int right = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (mino.shape[j][i] == BLOCK && right < j) {
                right = j;
            }
        }
    }
    //一番右のミノを返す
    // printf("right = %d\n", right);
    return right;
}
//降りれるかどうかを判定する関数
int can_Down(Mino mino) {
    int bottom;
    bottom = under_mino(mino);

    // printf("Can_down\n");
    //一番したに到達しているなら降りれないと判定する
    if (mino.y + bottom <= 1) {
        return FALSE;
    }

    //下にミノがあるのかを調べる
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            //ミノがあって下にミノがあるなら降りれないと判定
            //一番下がミノでないときの判定
            if (mino.shape[i][j] != BLOCK && j + 1 < 4 &&
                mino.shape[i][j + 1] == BLOCK &&
                grid[mino.x + i][mino.y + j] == BLOCK) {
                // printf("i = %d, j = %d, x = %d, y = %d\n", i, j, x, y);
                //降りれないと判定
                return FALSE;
            }
            //一番したがミノであるときの判定
            else if (bottom == 0 && bottom == j && mino.shape[i][j] == BLOCK &&
                     grid[mino.x + i][mino.y + j - 1] == BLOCK) {
                return FALSE;
            }
        }
    }
    // printf("can_Down's y = %d\n", mino.y);
    return TRUE;
}
//左に行けるかどうかを判定する関数
int can_Left(Mino mino) {
    int left = left_mino(mino);
    if (mino.x + left <= 0) return FALSE;
    for (int j = 0; j < 4; j++) {
        // printf("mino.x %d i %d\n", mino.x, i);
        if (mino.shape[left][j] == BLOCK &&
            grid[mino.x + left - 1][mino.y + j] == BLOCK) {
            //printf("mino.x %d i %d", mino.x, i);
            return FALSE;
        }
    }
    // printf("left is Ok\n");
    return TRUE;
}
//右に行けるかどうかを判定する関数
int can_right(Mino mino) {
    int right = right_mino(mino);
    // printf("right \n");
    if (mino.x + right + 1 >= 10) return FALSE;
    for (int j = 0; j < 4; j++) {
        if (mino.shape[right][j] == BLOCK &&
            grid[mino.x + right + 1][mino.y + j] == BLOCK) {
            return FALSE;
        }
    }
    return TRUE;
}
Mino minoRotateSetup(Mino mino) {
    //初期化
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mino.rot_shape[i][j] = NONBLOCK;
        }
    }
    //回転できたとみなして回転したときの座標を入れる
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mino.rot_shape[3 - j][i] = mino.shape[i][j];
            // if(mino.rot_shape[i][j] == 1)printf("rot_shape x %d y %d\n", i,
            // j);
        }
    }
    return mino;
}
//右回転に関する判定
int can_right_rotate(Mino mino) {
    //回転したときのx, y座標
    int rotx, roty;
    int flag = FALSE;
    mino = minoRotateSetup(mino);
    // printf("mino.x = %d\n", mino.x);
    //回転した先が回転元と被らないように一時的に消す
    for (int i = left_mino(mino); i < 4; i++) {
        for (int j = under_mino(mino); j < 4; j++) {
            if (grid[mino.x + i][mino.y + j] == BLOCK && mino.shape[i][j] == BLOCK) {
                grid[mino.x + i][mino.y + j] = NONBLOCK;
            }
        }
    }
    //回転できるかどうかの処理
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            //もしミノがあれば
            if (mino.rot_shape[i][j] == BLOCK) {
                //回転したときのx、y座標
                rotx = mino.x + i;
                roty = mino.y + j;
                // printf("before return rotx, roty %d %d\n", rotx, roty);
                //もし回転したときに上にこしたり、左右を超えたり、ミノがあれば回転できないと判断
                if (rotx <= 0 || rotx >= 10 || roty >= 21 ||
                    grid[rotx][roty] == BLOCK) {
                    // printf("rotx %d roty %d, i %d, j %d\n", rotx, roty, i,
                    // j); フラグを立てる
                    flag = TRUE;
                }
            }
        }
    }
    //もしフラグが立っていたら
    if (flag == TRUE) {
        //消えているところを元に戻して戻れないと返す
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                // printf("mino.x + i %d mino.y + j %d\n",mino.x + i,mino.y +
                // j);
                if (mino.shape[i][j] == BLOCK) {
                    grid[mino.x + i][mino.y + j] = mino.shape[i][j];
                }
            }
        }
        return FALSE;
    }
    return TRUE;
}
//ライン消去に関する関数
int line_erase(Mino mino) {
    int count;  //マス目ごとの数を計算する変数
    int erase_count = 0; //空白になったところの数を数える変数
    int tmp;//１つずつ減らす際にerase_countを保存しておくための変数
    int non_space;  //空白の行かどうかを調べる変数
    int fall_space; //落とした行が空白の場合、もう一回調べるための変数
    int delete; //１回落としたときに余計にループをしないためにerase_countを減らすための変数

    //printf("lien_erase\n");
    for (int j = 1; j < 20; j++) {
        //カウントを初期化する
        count = 0;
        for (int i = 0; i < 10; i++) {
            //もしマス目があるなら+1する
            if (grid[i][j] == 1) count++;
        }
        //もし行が全て埋まっているなら行を消し、消えた行をカウントする
        if (count == 10) {
            for (int i = 0; i < 10; i++) {
                grid[i][j] = NONBLOCK;
            }
            //消した分をカウントする
            erase_count++;
        }
    }
    //erase_countの数が消えないように一時的に保存する
    tmp = erase_count;
    //もし消えた行があるなら
    if (erase_count > 0) {
        // printf("erase_count %d & y %d\n", erase_count, y);
        //空白の行の上にあるミノをすべて消えた分落とす
        for (int i = 1; i < 21; i++) {
            //初期化する
            non_space = FALSE;  
            fall_space = FALSE;
            delete = FALSE; 
            //printf("&& erase_count %d flag %d\n", erase_count, flag2);
            //空白の行を調べる。空白なら落とし、違うなら次のループを飛ばす
            for (int j = 0; j < 10; j++) {
                if (grid[j][i] == BLOCK) non_space = TRUE;
            }
            //空白の行なら上にあるミノを全て１つずつ落としていく。
            for (int j = 0; j < 10; j++) {
                if (non_space == FALSE) {
                    //printf("shift i %d\n", i);
                    grid[j][i] = grid[j][i + 1];
                    for (int k = i + 1; k < 20; k++) {
                        grid[j][k] = grid[j][k + 1];
                    }
                    //１つ減らしたということを変数に保存しておく
                    delete = TRUE;
                }
            }
            //もし上のループを通っているなら空白の行は１つなくなっているので１つ減らす
            if (delete == TRUE) erase_count--;
            //落としてきた行がまた空白の行かどうかを調べる
            for (int j = 0; j < 10; j++) {
                if (grid[j][i] == BLOCK) fall_space = TRUE;
            }
            //もし空白の行ならもう一回ループをする
            if (fall_space == FALSE) {
                i--;
            }
            //もし空白の分落としたならループから抜け出す。
            if (erase_count == 0) break;
        }
    }
    //元に戻すして消えた分を返す
    erase_count = tmp;
    return erase_count;
}
//ミノを落とす処理
Mino down_mino(Mino mino) {
    //ミノを下に落とす計算
    for (int i = 0; i < 4; i++) {
        for (int j = under_mino(mino); j < 4; j++) {
            // printf("mino.x %d i %d mino.y %d j %d\n",mino.x, i, mino.y, j);
            if (grid[mino.x + i][mino.y + j] == BLOCK &&
                mino.shape[i][j] == BLOCK) {  //ミノがあって形が一致しているなら消す
                // printf("erase x = %d, y = %d\n", x + i, y + j);
                //動かすところを消す
                grid[mino.x + i][mino.y + j] = NONBLOCK;
            }
        }
    }
    //消してから１つ下げる
    mino.y += mino.dy;
    //下にミノを入れるための処理
    for (int i = left_mino(mino); i < 4; i++) {
        for (int j = under_mino(mino); j < 4; j++) {
            // printf("mino.x %d i %d mino.y %d j %d\n",mino.x, i, mino.y, j);
            if (grid[mino.x + i][mino.y + j] == NONBLOCK) {
                grid[mino.x + i][mino.y + j] = mino.shape[i][j];
            }
        }
    }
    return mino;
}
//左に動かす処理
Mino move_left(Mino mino) {
    for (int i = left_mino(mino); i < 4; i++) {
        for (int j = under_mino(mino); j < 4; j++) {
            if (grid[mino.x + i][mino.y + j] == BLOCK &&
                mino.shape[i][j] == BLOCK) {  //ミノがあって形が一致しているなら消す
                // printf("erase x = %d, y = %d\n", x + i, y + j);
                //動かすところを消す
                grid[mino.x + i][mino.y + j] = NONBLOCK;
            }
        }
    }
    mino.x--;
    //左に言ったところを描く
    for (int i = left_mino(mino); i < 4; i++) {
        for (int j = under_mino(mino); j < 4; j++) {
            if (grid[mino.x + i][mino.y + j] == NONBLOCK) {
                grid[mino.x + i][mino.y + j] = mino.shape[i][j];
            }
        }
    }
    return mino;
}
//右に動かす処理
Mino move_right(Mino mino) {
    //一番左のミノと一番下のミノから消す処理をする
    for (int i = left_mino(mino); i < 4; i++) {
        for (int j = under_mino(mino); j < 4; j++) {
            if (grid[mino.x + i][mino.y + j] == BLOCK &&
                mino.shape[i][j] == BLOCK) {  //ミノがあって形が一致しているなら消す
                // printf("erase x = %d, y = %d\n", x + i, y + j);
                //動かすところを消す
                grid[mino.x + i][mino.y + j] = NONBLOCK;
            }
        }
    }
    //右に１つずらす
    mino.x++;
    //ずらしたところにミノを描く
    for (int i = left_mino(mino); i < 4; i++) {
        for (int j = under_mino(mino); j < 4; j++) {
            if (grid[mino.x + i][mino.y + j] == NONBLOCK) {
                grid[mino.x + i][mino.y + j] = mino.shape[i][j];
            }
        }
    }
    return mino;
}
//右回転の処理をする
Mino rotate_right(Mino mino) {
    //回転した後の形を元の形に戻す
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mino.shape[i][j] = mino.rot_shape[i][j];
        }
    }
    /*
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            if(shape[i][j] == 1){
                printf("shape i %d j %d\n", i, j);
            }
        }
    }*/
    //ミノをグリッドに入れる
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (grid[mino.x + i][mino.y + j] == NONBLOCK) {
                grid[mino.x + i][mino.y + j] = mino.shape[i][j];
            }
        }
    }
    return mino;
}
//色とフィールドを描く関数
void field_color() {
    //色の設定1:白 それ以外: グレー
    for (int i = 0; i < 10; i++) {
        for (int j = 1; j < 21; j++) {
            if (grid[i][j] == 1) {
                HgSetFillColor(HG_WHITE);
            } else {
                HgSetFillColor(HG_GRAY);
            }
            HgBoxFill(i * 30, j * 30, 30, 30, 1);
        }
    }
}
//ゲームオーバーの処理
int gameOver(void) {
    hgevent *event;
    field_color();
    //赤色でゲームオーバーと描く
    HgSetColor(HG_RED);
    HgSetFont(HG_T, 30);
    HgText(WINDOWSIZE / 4, WINDOWSIZE, "GAME OVER");
    //下の空白にコンティニューすルカどうかを描く
    HgSetColor(HG_BLACK);
    HgSetFont(HG_T, 9);
    HgText(5, 5, "60秒後に閉じます。コンティニュー？ PRESS c");
    HgSetEventMask(HG_KEY_DOWN);
    HgSetAlarmTimer(60);
    //６０秒たつかcが押されるまでループする
    for (;;) {
        event = HgEvent();
        if (event->ch == 'c') {
            return TRUE;
        }
        if (event->type == HG_TIMER_FIRE) {
            return FALSE;
        }
    }
    return 0;
}
//スコアを描く処理
void score_color(int erase_count) {
    //画面を消す
    HgClear();
    //空白のところにスコアを描く
    HgSetColor(HG_BLACK);
    HgSetFont(HG_T, 15);
    HgText(200, 5, "score is %d", erase_count * 100);
}
