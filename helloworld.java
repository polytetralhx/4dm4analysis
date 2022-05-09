import java.util.Scanner;
// I have figured out how the fuck to torture myself
class HelloWorld {

    public String getHelloWorld() {
        return "Hello World";
    }

    public static void main(String[] args) {
        int a = 2 + 3;
        System.out.println("Hello World");
        System.out.println(a);
    }

}

class AnotherHelloWorld {

    public static String isBetterThanVBA(int betterthanvba) {
        if (betterthanvba == 1) {
            return "Java is better than VBA";
        }
        return "Java is more fucked than VBA";
    }

    public static void main(String[] args) {
        System.out.println("Is Java better than VBA ? (0 / 1)");
        Scanner yes = new Scanner(System.in);
        int betterthanvba = yes.nextInt();
        System.out.println(isBetterThanVBA(betterthanvba));
        yes.close();
    }
}