using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Text;


public static class App
{
  public static void Main()
  {
    for (int i = 0; i < 1000; ++i)
    {
      var img = new CaptchaImage();
      img.RenderImage().Save(".\\images\\" + img.Text + ".bmp");
    }
  }
}


public class CaptchaImage
{
	public enum FontWarpFactor
	{
		None,
		Low,
		Medium,
		High,
		Extreme
	}
	public enum BackgroundNoiseLevel
	{
		None,
		Low,
		Medium,
		High,
		Extreme
	}
	public enum LineNoiseLevel
	{
		None,
		Low,
		Medium,
		High,
		Extreme
	}
	private int _height;
	private int _width;
	private Random _rand;
	private DateTime _generatedAt;
	private string _randomText;
	private int _randomTextLength;
	private string _randomTextChars;
	private string _fontFamilyName;
	private CaptchaImage.FontWarpFactor _fontWarp;
	private CaptchaImage.BackgroundNoiseLevel _backgroundNoise;
	private CaptchaImage.LineNoiseLevel _lineNoise;
	private string _guid;
	private string _fontWhitelist;
	private string[] RandomFontFamilyFF;
	public string UniqueId
	{
		get
		{
			return this._guid;
		}
	}
	public DateTime RenderedAt
	{
		get
		{
			return this._generatedAt;
		}
	}
	public string Font
	{
		get
		{
			return this._fontFamilyName;
		}
		set
		{
			try
			{
				Font font = new Font(value, 12f);
				this._fontFamilyName = value;
				font.Dispose();
			}
			catch (Exception)
			{
				this._fontFamilyName = FontFamily.GenericSerif.Name;
			}
		}
	}
	public CaptchaImage.FontWarpFactor FontWarp
	{
		get
		{
			return this._fontWarp;
		}
		set
		{
			this._fontWarp = value;
		}
	}
	public CaptchaImage.BackgroundNoiseLevel BackgroundNoise
	{
		get
		{
			return this._backgroundNoise;
		}
		set
		{
			this._backgroundNoise = value;
		}
	}
	public CaptchaImage.LineNoiseLevel LineNoise
	{
		get
		{
			return this._lineNoise;
		}
		set
		{
			this._lineNoise = value;
		}
	}
	public string TextChars
	{
		get
		{
			return this._randomTextChars;
		}
		set
		{
			this._randomTextChars = value;
			this._randomText = this.GenerateRandomText();
		}
	}
	public int TextLength
	{
		get
		{
			return this._randomTextLength;
		}
		set
		{
			this._randomTextLength = value;
			this._randomText = this.GenerateRandomText();
		}
	}
	public string Text
	{
		get
		{
			return this._randomText;
		}
	}
	public int Width
	{
		get
		{
			return this._width;
		}
		set
		{
			if (value <= 60)
			{
				throw new ArgumentOutOfRangeException("width", value, "width must be greater than 60.");
			}
			this._width = value;
		}
	}
	public int Height
	{
		get
		{
			return this._height;
		}
		set
		{
			if (value <= 30)
			{
				throw new ArgumentOutOfRangeException("height", value, "height must be greater than 30.");
			}
			this._height = value;
		}
	}
	public string FontWhitelist
	{
		get
		{
			return this._fontWhitelist;
		}
		set
		{
			this._fontWhitelist = value;
		}
	}
	public CaptchaImage()
	{
		this._rand = new Random();
		this._fontWarp = CaptchaImage.FontWarpFactor.Low;
		this._backgroundNoise = CaptchaImage.BackgroundNoiseLevel.Low;
		this._lineNoise = CaptchaImage.LineNoiseLevel.None;
		this._width = 180;
		this._height = 50;
		this._randomTextLength = 5;
		this._randomTextChars = "ACDEFGHJKLNPQRTUVXYZ2346789";
		this._fontFamilyName = "";
		this._fontWhitelist = "arial;arial black;comic sans ms;courier new;estrangelo edessa;franklin gothic medium;georgia;lucida console;lucida sans unicode;mangal;microsoft sans serif;palatino linotype;sylfaen;tahoma;times new roman;trebuchet ms;verdana";
		this._randomText = this.GenerateRandomText();
		this._generatedAt = DateTime.Now;
		this._guid = Guid.NewGuid().ToString();
	}
	public Bitmap RenderImage()
	{
		return this.GenerateImagePrivate();
	}
	private string RandomFontFamily()
	{
		if (this.RandomFontFamilyFF == null)
		{
			this.RandomFontFamilyFF = this._fontWhitelist.Split(new char[]
			{
				';'
			});
		}
		return this.RandomFontFamilyFF[this._rand.Next(0, this.RandomFontFamilyFF.Length)];
	}
	private string GenerateRandomText()
	{
		StringBuilder stringBuilder = new StringBuilder(this._randomTextLength);
		int length = this._randomTextChars.Length;
		checked
		{
			int num = this._randomTextLength - 1;
			for (int i = 0; i <= num; i++)
			{
				stringBuilder.Append(this._randomTextChars.Substring(this._rand.Next(length), 1));
			}
			return stringBuilder.ToString();
		}
	}
	private PointF RandomPoint(int xmin, int xmax, ref int ymin, ref int ymax)
	{
		PointF result = new PointF((float)this._rand.Next(xmin, xmax), (float)this._rand.Next(ymin, ymax));
		return result;
	}
	private PointF RandomPoint(Rectangle rect)
	{
		int top = rect.Top;
		int bottom = rect.Bottom;
		return this.RandomPoint(rect.Left, rect.Width, ref top, ref bottom);
	}
	private GraphicsPath TextPath(string s, Font f, Rectangle r)
	{
		StringFormat stringFormat = new StringFormat();
		stringFormat.Alignment = StringAlignment.Near;
		stringFormat.LineAlignment = StringAlignment.Near;
		GraphicsPath graphicsPath = new GraphicsPath();
		graphicsPath.AddString(s, f.FontFamily, (int)f.Style, f.Size, r, stringFormat);
		return graphicsPath;
	}
	private Font GetFont()
	{
		string text = this._fontFamilyName;
		if (string.Compare(text, "", false) == 0)
		{
			text = this.RandomFontFamily();
		}
		float emSize;
		switch (this.FontWarp)
		{
		case CaptchaImage.FontWarpFactor.Low:
			emSize = (float)Convert.ToInt32((double)this._height * 0.8);
			break;

		case CaptchaImage.FontWarpFactor.Medium:
			emSize = (float)Convert.ToInt32((double)this._height * 0.85);
			break;

		case CaptchaImage.FontWarpFactor.High:
			emSize = (float)Convert.ToInt32((double)this._height * 0.9);
			break;

		case CaptchaImage.FontWarpFactor.Extreme:
			emSize = (float)Convert.ToInt32((double)this._height * 0.95);
			break;
		case CaptchaImage.FontWarpFactor.None:
                default:
			emSize = (float)Convert.ToInt32((double)this._height * 0.7);
			break;
		}
		return new Font(text, emSize, FontStyle.Bold);
	}
	private Bitmap GenerateImagePrivate()
	{
		Font font = null;
		Bitmap bitmap = new Bitmap(this._width, this._height, PixelFormat.Format32bppArgb);
		Graphics graphics = Graphics.FromImage(bitmap);
		graphics.SmoothingMode = SmoothingMode.AntiAlias;
		Rectangle rect = new Rectangle(0, 0, this._width, this._height);
		Brush brush = new SolidBrush(Color.White);
		graphics.FillRectangle(brush, rect);
		int num = 0;
		double num2 = (double)this._width / (double)this._randomTextLength;
		string randomText = this._randomText;
		int i = 0;
		int length = randomText.Length;
		while (i < length)
		{
			char value = randomText[i];
			font = this.GetFont();
			Rectangle rectangle = new Rectangle(Convert.ToInt32((double)num * num2), 0, Convert.ToInt32(num2), this._height);
			GraphicsPath graphicsPath = this.TextPath(value.ToString(), font, rectangle);
			this.WarpText(graphicsPath, rectangle);
			brush = new SolidBrush(Color.Black);
			graphics.FillPath(brush, graphicsPath);
			checked
			{
				num++;
				i++;
			}
		}
		this.AddNoise(graphics, rect);
		this.AddLine(graphics, rect);
		font.Dispose();
		brush.Dispose();
		graphics.Dispose();
		return bitmap;
	}
	private void WarpText(GraphicsPath textPath, Rectangle rect)
	{
		float num;
		float num2;
		switch (this._fontWarp)
		{
		case CaptchaImage.FontWarpFactor.Low:
			num = 6f;
			num2 = 1f;
			break;

		case CaptchaImage.FontWarpFactor.Medium:
			num = 5f;
			num2 = 1.3f;
			break;

		case CaptchaImage.FontWarpFactor.High:
			num = 4.5f;
			num2 = 1.4f;
			break;

		case CaptchaImage.FontWarpFactor.Extreme:
			num = 4f;
			num2 = 1.5f;
			break;
		case CaptchaImage.FontWarpFactor.None:
                default:
			return;

		}
                var r = num + num2;
                System.Diagnostics.Trace.WriteLine(r.ToString());
		RectangleF srcRect = new RectangleF(Convert.ToSingle(rect.Left), 0f, Convert.ToSingle(rect.Width), (float)rect.Height);
		int num3 = Convert.ToInt32((float)rect.Height / num);
		int num4 = Convert.ToInt32((float)rect.Width / num);
		checked
		{
			int num5 = rect.Left - Convert.ToInt32(unchecked((float)num4 * num2));
			int num6 = rect.Top - Convert.ToInt32(unchecked((float)num3 * num2));
			int num7 = rect.Left + rect.Width + Convert.ToInt32(unchecked((float)num4 * num2));
			int num8 = rect.Top + rect.Height + Convert.ToInt32(unchecked((float)num3 * num2));
			if (num5 < 0)
			{
				num5 = 0;
			}
			if (num6 < 0)
			{
				num6 = 0;
			}
			if (num7 > this.Width)
			{
				num7 = this.Width;
			}
			if (num8 > this.Height)
			{
				num8 = this.Height;
			}
			int arg_155_1 = num5;
			int arg_155_2 = num5 + num4;
			int num9 = num6 + num3;
			PointF pointF = this.RandomPoint(arg_155_1, arg_155_2, ref num6, ref num9);
			int arg_16E_1 = num7 - num4;
			int arg_16E_2 = num7;
			num9 = num6 + num3;
			PointF pointF2 = this.RandomPoint(arg_16E_1, arg_16E_2, ref num6, ref num9);
			int arg_184_1 = num5;
			int arg_184_2 = num5 + num4;
			num9 = num8 - num3;
			PointF pointF3 = this.RandomPoint(arg_184_1, arg_184_2, ref num9, ref num8);
			int arg_19B_1 = num7 - num4;
			int arg_19B_2 = num7;
			num9 = num8 - num3;
			PointF pointF4 = this.RandomPoint(arg_19B_1, arg_19B_2, ref num9, ref num8);
			PointF[] destPoints = new PointF[]
			{
				pointF,
				pointF2,
				pointF3,
				pointF4
			};
			Matrix matrix = new Matrix();
			matrix.Translate(0f, 0f);
			textPath.Warp(destPoints, srcRect, matrix, WarpMode.Perspective, 0f);
		}
	}
	private void AddNoise(Graphics graphics1, Rectangle rect)
	{
		int num;
		int num2;
		switch (this._backgroundNoise)
		{
		case CaptchaImage.BackgroundNoiseLevel.None:
		default:
			return;

		case CaptchaImage.BackgroundNoiseLevel.Low:
			num = 30;
			num2 = 40;
			break;

		case CaptchaImage.BackgroundNoiseLevel.Medium:
			num = 18;
			num2 = 40;
			break;

		case CaptchaImage.BackgroundNoiseLevel.High:
			num = 16;
			num2 = 39;
			break;

		case CaptchaImage.BackgroundNoiseLevel.Extreme:
			num = 12;
			num2 = 38;
			break;
		}
		SolidBrush solidBrush = new SolidBrush(Color.Black);
		int maxValue = Convert.ToInt32((double)Math.Max(rect.Width, rect.Height) / (double)num2);
		int arg_89_0 = 0;
		checked
		{
			int num3 = Convert.ToInt32((double)(rect.Width * rect.Height) / (double)num);
			for (int i = arg_89_0; i <= num3; i++)
			{
				graphics1.FillEllipse(solidBrush, this._rand.Next(rect.Width), this._rand.Next(rect.Height), this._rand.Next(maxValue), this._rand.Next(maxValue));
			}
			solidBrush.Dispose();
		}
	}
	private void AddLine(Graphics graphics1, Rectangle rect)
	{
		int num;
		float width;
		int num2;
		switch (this._lineNoise)
		{
		case CaptchaImage.LineNoiseLevel.None:
		default:
			return;

		case CaptchaImage.LineNoiseLevel.Low:
			num = 4;
			width = Convert.ToSingle((double)this._height / 31.25);
			num2 = 1;
			break;

		case CaptchaImage.LineNoiseLevel.Medium:
			num = 5;
			width = Convert.ToSingle((double)this._height / 27.7777);
			num2 = 1;
			break;

		case CaptchaImage.LineNoiseLevel.High:
			num = 3;
			width = Convert.ToSingle((double)this._height / 25.0);
			num2 = 2;
			break;

		case CaptchaImage.LineNoiseLevel.Extreme:
			num = 3;
			width = Convert.ToSingle((double)this._height / 22.7272);
			num2 = 3;
			break;
		}
		checked
		{
			PointF[] array = new PointF[num + 1];
			Pen pen = new Pen(Color.Black, width);
			int arg_B8_0 = 1;
			int num3 = num2;
			for (int i = arg_B8_0; i <= num3; i++)
			{
				int arg_C0_0 = 0;
				int num4 = num;
				for (int j = arg_C0_0; j <= num4; j++)
				{
					array[j] = this.RandomPoint(rect);
				}
				graphics1.DrawCurve(pen, array, 1.75f);
			}
			pen.Dispose();
		}
	}
}
