import java.io.File;
import java.io.IOException;
import java.io.PrintStream;



public class Main {
	
	public static void main(String[] args) throws IOException, InterruptedException{
		
		/* Configuration du Hadoop */
		String filepath = "/cal/homes/johayon/INF727";
		String filename = "INPUT2.txt";
		int Number_Worker = 4;
		String liste_ordi = "liste_ordi.txt";
		
		/* enleve les log de sfl4j */
		File logs = new File(filepath + "/log.txt");
		System.setErr(new PrintStream(logs));
		
		Shavadoop_Master test = new Shavadoop_Master(filepath,filename,Number_Worker,liste_ordi);
		int startTime = (int) System.currentTimeMillis();
		test.getMachine();
		System.out.println("----------------------------     SHAVADOOP BEGINS ----------------------------------- \n" );
		test.Splitfile();
		test.Mapper();
		test.UMlocation();
		//test.dico();
		test.shuffle();
		test.reducer();
		test.results();
		
		
		System.out.println("----------------------------     SHAVADOOP ENDS -----------------------------------" );
		int endTime = (int) System.currentTimeMillis();
		System.out.println("Temps final = " + new Integer(endTime - startTime).toString() + " ms" );
	}
	

}
